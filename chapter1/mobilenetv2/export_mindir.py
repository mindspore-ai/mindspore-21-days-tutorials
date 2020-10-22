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
export .mindir format file for MindSpore Lite reasoning.
"""
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
from src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export .mindir model file in the training side.')
    parser.add_argument('--platform', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'],
                        help='run platform, only support CPU, GPU and Ascend')
    parser.add_argument('--ckpt_path', type=str, required=True, default='./mobilenetV2-10_1562.ckpt',
                        help='Pretrained checkpoint path')
    parser.add_argument('--mindir_name', type=str, default='mobilenetv2.mindir',
                        help='.mindir model file name')
    args = parser.parse_args()
    backbone_net = MobileNetV2Backbone()
    head_net = MobileNetV2Head(input_channel=backbone_net.out_channels,
                               num_classes=10,
                               activation="Softmax")
    mobilenet = mobilenet_v2(backbone_net, head_net)
    # return a parameter dict for model
    param_dict = load_checkpoint(args.ckpt_path)
    # load the parameter into net
    load_param_into_net(mobilenet, param_dict)
    input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
    export(mobilenet, Tensor(input), file_name=args.mindir_name, file_format='MINDIR')

