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

'''
Bert finetune and evaluation script.
'''

import os
import argparse
from src.bert_for_finetune import BertFinetuneCell, BertCLS
from src.finetune_eval_model import BertCLSModel
from src.finetune_eval_config import optimizer_cfg, bert_net_cfg, cloud_cfg
from src.dataset import create_classification_dataset
from src.assessment_method import Accuracy, F1, MCC, Spearman_Correlation
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, BertLearningRate
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from convert_example import convert_text

_cur_dir = os.getcwd()

os.environ['MINDSPORE_HCCL_CONFIG_PATH'] = os.getenv('RANK_TABLE_FILE')
job_id = os.getenv('JOB_ID')
job_id = job_id if job_id != "" else "default"
device_id = int(os.getenv('DEVICE_ID'))
device_num = int(os.getenv('RANK_SIZE'))
global_rank_id = int(os.getenv('RANK_ID').split('-')[-1])
rank_id = device_id + global_rank_id * 8


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
        print(os.listdir("/cache/data"))
        print("===save flag===")
    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                       end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="tnews",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertFinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks)

def eval_result_print(assessment_method="accuracy", callback=None):
    """ print eval result """
    if assessment_method == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
    elif assessment_method == "f1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
    elif assessment_method == "mcc":
        print("MCC {:.6f} ".format(callback.cal()))
    elif assessment_method == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback.cal()[0]))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

def do_eval(dataset=None, network=None, num_class=2, assessment_method="accuracy", load_checkpoint_path=""):
    """ do eval """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(bert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(net_for_pretraining, param_dict)
    model = Model(net_for_pretraining)

    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "f1":
        callback = F1(False, num_class)
    elif assessment_method == "mcc":
        callback = MCC()
    elif assessment_method == "spearman_correlation":
        callback = Spearman_Correlation()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in dataset.create_dict_iterator():
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
        callback.update(logits, label_ids)
    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")


def do_predict(args_opt):
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
    net = BertCLSModel(bert_net_cfg, False, len(labels))
    net.set_train(False)
    param_dict = load_checkpoint(cloud_cfg.finetune_ckpt)
    load_param_into_net(net, param_dict)
    model = Model(net)
    input_features = convert_text(args_opt.predict, cloud_cfg.vocab_file, bert_net_cfg.seq_length)
    input_ids = Tensor(input_features.input_ids, mstype.int32)
    input_mask = Tensor(input_features.input_mask, mstype.int32)
    token_type_id = Tensor(input_features.segment_ids, mstype.int32)
    logits = model.predict(input_ids, input_mask, token_type_id)
    print("==============================================================")
    cls = logits.asnumpy().argmax()
    print("output: ", cls)
    print("cls: ", labels[cls])
    print("==============================================================")

def run_classifier():
    """run classifier task"""
    parser = argparse.ArgumentParser(description="run classifier")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument("--assessment_method", type=str, default="Accuracy",
                        choices=["Mcc", "Spearman_correlation", "Accuracy", "F1"],
                        help="assessment_method including [Mcc, Spearman_correlation, Accuracy, F1],\
                             default is Accuracy")
    parser.add_argument("--do_train", type=str, default="false", choices=["true", "false"],
                        help="Enable train, default is false")
    parser.add_argument("--do_eval", type=str, default="false", choices=["true", "false"],
                        help="Enable eval, default is false")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--epoch_num", type=int, default="1", help="Epoch number, default is 1.")
    parser.add_argument("--num_class", type=int, default="2", help="The number of class, default is 2.")
    parser.add_argument("--train_data_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--save_finetune_checkpoint_path", type=str, default="/cache/train", help="Save checkpoint path")
    parser.add_argument("--load_pretrain_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--load_finetune_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_file_path", type=str, default="",
                        help="Schema path, it is better to use absolute path")
    parser.add_argument('--data_url', type=str, default='Ascend', help='data url')
    parser.add_argument('--train_url', type=str, default='Ascend', help='train url')
    parser.add_argument('--predict', type=str, default='', help='text to predict')
    args_opt = parser.parse_args()
    epoch_num = args_opt.epoch_num
    assessment_method = args_opt.assessment_method.lower()
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false" and args_opt.predict == "":
        raise ValueError("At least one of 'do_train' or 'do_eval'  or 'predict' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")

    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    if args_opt.predict != '':
        do_predict(args_opt)
        return
    else:
        sync_dataset(args_opt.data_url)

    netwithloss = BertCLS(bert_net_cfg, True, num_labels=args_opt.num_class, dropout_prob=0.1,
                          assessment_method=assessment_method)

    if args_opt.do_train.lower() == "true":
        ds = create_classification_dataset(batch_size=optimizer_cfg.batch_size, repeat_count=1,
                                           assessment_method=assessment_method,
                                           data_file_path=args_opt.train_data_file_path,
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))
        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, epoch_num)

        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir,
                                                           ds.get_dataset_size(), epoch_num, "tnews")
        # Copy the checkpoints back to obs
        if rank_id % device_num == 0:
            import moxing as mox
            mox.file.copy_parallel("/cache/train", args_opt.train_url)

    if args_opt.do_eval.lower() == "true":
        ds = create_classification_dataset(batch_size=optimizer_cfg.batch_size, repeat_count=1,
                                           assessment_method=assessment_method,
                                           data_file_path=args_opt.eval_data_file_path,
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
        do_eval(ds, BertCLS, args_opt.num_class, assessment_method, load_finetune_checkpoint_path)

if __name__ == "__main__":
    run_classifier()
