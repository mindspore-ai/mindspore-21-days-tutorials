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
"""ResNet50 model eval with MindSpore"""
import argparse
import random
import numpy as np
import moxing as mox

import mindspore.context as context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import cfg
from src.resnet import resnet50
from src.dataset import create_dataset
from src.CrossEntropySmooth import CrossEntropySmooth

random.seed(1)
np.random.seed(1)


def resnet50_eval(args_opt):
    class_num = cfg.class_num
    local_data_path = '/cache/data'
    ckpt_file_slice = args_opt.checkpoint_path.split('/')
    ckpt_file = ckpt_file_slice[len(ckpt_file_slice)-1]
    local_ckpt_path = '/cache/'+ckpt_file

    # set graph mode and parallel mode
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)

    # data download
    print('Download data.')
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=local_data_path)
    mox.file.copy_parallel(src_url=args_opt.checkpoint_path, dst_url=local_ckpt_path)

    # create dataset
    dataset = create_dataset(dataset_path=local_data_path, do_train=False,
                             batch_size=cfg.batch_size)

    # load checkpoint into net
    net = resnet50(class_num=class_num)
    param_dict = load_checkpoint(local_ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # define loss and model
    if not cfg.use_label_smooth:
        cfg.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction='mean',
                              smooth_factor=cfg.label_smooth_factor,
                              num_classes=cfg.class_num)
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    # eval model
    res = model.eval(dataset)
    print("result:", res, "ckpt=", args_opt.checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet50 eval.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--checkpoint_path', required=True, type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--device_target', type=str, default='Ascend', help='Device target. Default: Ascend.')
    args_opt, unknown = parser.parse_known_args()

    resnet50_eval(args_opt)
    print('ResNet50 evaluation success!')
