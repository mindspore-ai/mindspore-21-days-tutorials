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

import argparse
import ast


def parse_args():
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--is_training', type=bool, default=True,
                        help='training is True, eval is False.')
    parser.add_argument('--platform', type=str, default="GPU", choices=("CPU", "GPU", "Ascend"),
                        help='run platform, only support CPU, GPU and Ascend')
    parser.add_argument('--dataset_path', type=str, default='../data/', help='Dataset path')
    parser.add_argument('--epoch_size', type=int, default=1, help='Train epoch size')
    parser.add_argument('--pretrain_ckpt', type=str, default="",
                        help='Pretrained checkpoint path for fine tune or incremental learning')
    parser.add_argument('--freeze_layer', type=str, default="", choices=["", "none", "backbone"],
                        help="freeze the weights of network from start to which layers")
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    args = parser.parse_args()
    return args
