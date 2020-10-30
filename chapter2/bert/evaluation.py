# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Bert evaluation script.
"""

import os
import sys
import argparse
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.evaluation_config import cfg, bert_net_cfg
from src.utils import BertNER, BertCLS, BertCLSModel
from src.CRF import postprocess
#from src.cluener_evaluation import submit
from src.finetune_config import tag_to_index
from convert_example import convert_text

import moxing as mox

os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')
job_id = os.getenv('JOB_ID')
job_id = job_id if job_id != "" else "default"
device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
global_rank_id = int(os.getenv('RANK_ID').split('-')[-1])
rank_id = device_id + global_rank_id * 8

class Accuracy():
    '''
    calculate accuracy
    '''
    def __init__(self):
        self.acc_num = 0
        self.total_num = 0
    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logit_id)
        self.total_num += len(labels)
        print("=========================accuracy is ", self.acc_num / self.total_num)

class F1():
    '''
    calculate F1 score
    '''
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
    def update(self, logits, labels):
        '''
        update F1 score
        '''
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        if cfg.use_crf:
            backpointers, best_tag_id = logits
            best_path = postprocess(backpointers, best_tag_id)
            logit_id = []
            for ele in best_path:
                logit_id.extend(ele)
        else:
            logits = logits.asnumpy()
            logit_id = np.argmax(logits, axis=-1)
            logit_id = np.reshape(logit_id, -1)
        pos_eva = np.isin(logit_id, [i for i in range(1, cfg.num_labels)])
        pos_label = np.isin(labels, [i for i in range(1, cfg.num_labels)])
        self.TP += np.sum(pos_eva&pos_label)
        self.FP += np.sum(pos_eva&(~pos_label))
        self.FN += np.sum((~pos_eva)&pos_label)

def get_dataset(batch_size=1, repeat_count=1, distribute_file=''):
    '''
    get dataset
    '''
    _ = distribute_file

    ds = de.TFRecordDataset([cfg.data_file], cfg.schema_file, columns_list=["input_ids", "input_mask",
                                                                            "segment_ids", "label_ids"])
    type_cast_op = C.TypeCast(mstype.int32)
    ds = ds.map(input_columns="segment_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.map(input_columns="label_ids", operations=type_cast_op)
    ds = ds.repeat(repeat_count)

    # apply shuffle operation
    buffer_size = 960
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def bert_predict(Evaluation):
    '''
    prediction function
    '''
    dataset = get_dataset(bert_net_cfg.batch_size, 1)
    if cfg.use_crf:
        net_for_pretraining = Evaluation(bert_net_cfg, False, num_labels=len(tag_to_index), use_crf=True,
                                         tag_to_index=tag_to_index, dropout_prob=0.0)
    else:
        net_for_pretraining = Evaluation(bert_net_cfg, False, num_labels)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)
    return model, dataset


def sync_dataset(data_url):
    import moxing as mox
    import sys
    import time
    sync_lock = "/tmp/copy_sync.lock"
    if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
        mox.file.copy_parallel(data_url, "/cache/data")
        print("===finish download datasets===")
        try:
            os.mknod(sync_lock)
        except:
            pass

        print("===save flag===")
    
    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)


def test_eval():
    '''
    evaluation function
    '''
    task_type = BertNER if cfg.task == "NER" else BertCLS
    model, dataset = bert_predict(task_type)
    if cfg.clue_benchmark:
        submit(model, cfg.data_file, bert_net_cfg.seq_length)
    else:
        callback = F1() if cfg.task == "NER" else Accuracy()
        columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(Tensor(data[i]))
            input_ids, input_mask, token_type_id, label_ids = input_data
            logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
            callback.update(logits, label_ids)
        print("==============================================================")
        if cfg.task == "NER":
            print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
            print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
            print("F1 {:.6f} ".format(2*callback.TP / (2*callback.TP + callback.FP + callback.FN)))
        else:
            print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                      callback.acc_num / callback.total_num))
        print("==============================================================")


def test_predict(text):
    '''
    prediction function
    '''

    labels = [
        {"label": "100", "label_desc": "news_story"},
        {"label": "101", "label_desc": "news_culture"},
        {"label": "102", "label_desc": "news_entertainment"},
        {"label": "103", "label_desc": "news_sports"},
        {"label": "104", "label_desc": "news_finance"},
        {"label": "106", "label_desc": "news_house"},
        {"label": "107", "label_desc": "news_car"},
        {"label": "108", "label_desc": "news_edu"},
        {"label": "109", "label_desc": "news_tech"},
        {"label": "110", "label_desc": "news_military"},
        {"label": "112", "label_desc": "news_travel"},
        {"label": "113", "label_desc": "news_world"},
        {"label": "114", "label_desc": "news_stock"},
        {"label": "115", "label_desc": "news_agriculture"},
        {"label": "116", "label_desc": "news_game"}
    ]
    bert_net_cfg.batch_size = 1
    net = BertCLSModel(bert_net_cfg, False, num_labels)
    net.set_train(False)
    param_dict = load_checkpoint(cfg.finetune_ckpt)
    load_param_into_net(net, param_dict)
    model = Model(net)
    input_features = convert_text(text, cfg.vocab_file, bert_net_cfg.seq_length)
    input_ids = Tensor(input_features.input_ids, mstype.int32)
    input_mask = Tensor(input_features.input_mask, mstype.int32)
    token_type_id = Tensor(input_features.segment_ids, mstype.int32)
    logits = model.predict(input_ids, input_mask, token_type_id)
    print("==============================================================")
    cls = logits.asnumpy().argmax()
    print("output: ", cls)
    print("cls: ", labels[cls])
    print("==============================================================")


parser = argparse.ArgumentParser(description='Bert eval')
parser.add_argument('--device_target', type=str, default='Ascend', help='Device target')
parser.add_argument('--data_url', type=str, default='Ascend', help='data url')
parser.add_argument('--train_url', type=str, default='Ascend', help='train url')
parser.add_argument('--predict', type=str, default='', help='text to predict')
args_opt = parser.parse_args()
if __name__ == "__main__":
    target = args_opt.device_target
    if target == "Ascend":
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=devid)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    print("start to run.", sys.argv)
    num_labels = cfg.num_labels
    if args_opt.predict != '':
        test_predict(args_opt.predict)
    else:
        sync_dataset(args_opt.data_url)
        test_eval()
