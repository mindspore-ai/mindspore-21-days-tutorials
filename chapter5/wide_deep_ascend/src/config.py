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
""" config. """
import argparse


def argparse_init():
    """
    argparse_init
    """
    parser = argparse.ArgumentParser(description='WideDeep')
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                        help="device where the code will be implemented. (Default: Ascend)")
    parser.add_argument("--data_url", type=str, required=True, help="mindrecord dataset location")
    parser.add_argument('--train_url', required=True, help='Location of training outputs.')
    parser.add_argument("--epochs", type=int, default=1, help="Total train epochs")
    parser.add_argument("--batch_size", type=int, default=16000, help="Training batch size.")
    parser.add_argument("--field_size", type=int, default=39, help="The number of features.")
    parser.add_argument("--ckpt_url", type=str, default="/21days-wide-deep/ckpt/", help="The location of the checkpoint file.")
    parser.add_argument("--dataset_type", type=str, default="mindrecord", help="tfrecord/mindrecord/hd5")
    return parser


def data_argparse_init():
    """
        data processing argparse_init
    """
    parser = argparse.ArgumentParser(description="Process dataset")
    parser.add_argument("--data_file", type=str, default="mini_demo.txt",
                        help='the 1 percent of the raw data file to be processed')
    parser.add_argument("--dense_dim", type=int, default=13, help='The number of your continues fields')
    parser.add_argument("--slot_dim", type=int, default=26,
                        help='The number of your sparse fields, it can also be called catelogy features.')
    parser.add_argument("--threshold", type=int, default=100,
                        help='Word frequency below this will be regarded as OOV. It aims to reduce the vocab size')
    parser.add_argument("--train_line_count", type=int, default=458405, help='The number of examples in your dataset')
    parser.add_argument("--skip_id_convert", type=int, default=0, choices=[0, 1],
                        help='Skip the id convert, regarding the original id as the final id.')
    return parser


class WideDeepConfig():
    """
    WideDeepConfig
    """

    def __init__(self):
        self.device_target = "Ascend"
        self.data_path = "/cache/mydata/"
        self.full_batch = False
        self.epochs = 1
        self.batch_size = 16000
        self.field_size = 39
        self.vocab_size = 200000
        self.emb_dim = 80
        self.deep_layer_dim = [1024, 512, 256, 128]
        self.deep_layer_act = 'relu'
        self.weight_bias_init = ['normal', 'normal']
        self.emb_init = 'normal'
        self.init_args = [-0.01, 0.01]
        self.dropout_flag = False
        self.keep_prob = 1.0
        self.l2_coef = 8e-5

        self.output_path = "./output"
        self.eval_file_name = "eval.log"
        self.loss_file_name = "loss.log"
        self.ckpt_path = "/cache/myckpt/"
        self.stra_ckpt = './checkpoints/strategy.ckpt'
        self.host_device_mix = 0
        self.dataset_type = "mindrecord"
        self.parameter_server = 0
        self.field_slice = False
        self.manual_shape = None
        self.data_url = ""
        self.train_url = ""
        self.ckpt_url = ""

        self.data_file = "mini_demo.txt"
        self.dense_dim = 13
        self.slot_dim =26
        self.threshold = 100
        self.train_line_count = 458405
        self.skip_id_convert = 0

    def argparse_init(self):
        """
        argparse_init
        """
        parser = argparse_init()
        args, _ = parser.parse_known_args()
        self.device_target = args.device_target
        self.data_url = args.data_url
        self.train_url = args.train_url
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.field_size = args.field_size
        self.ckpt_url = args.ckpt_url
        self.dataset_type = args.dataset_type

    def data_argparse_init(self):
        """
        data_argparse_init
        """
        parser = data_argparse_init()
        args, _ = parser.parse_known_args()
        self.data_file = args.data_file
        self.dense_dim = args.dense_dim
        self.slot_dim = args.slot_dim
        self.threshold = args.threshold
        self.train_line_count = args.train_line_count
        self.skip_id_convert = args.skip_id_convert


