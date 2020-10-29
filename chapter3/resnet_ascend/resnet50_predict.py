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
"""ResNet50 model predict with MindSpore"""
import os
import argparse
import random
import cv2
import numpy as np
import moxing as mox

import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from config import cfg
from resnet import resnet50

random.seed(1)
np.random.seed(1)

label_list = ["Agaricus双孢蘑菇,伞菌目,蘑菇科,蘑菇属,广泛分布于北半球温带,无毒",
              "Amanita毒蝇伞,伞菌目,鹅膏菌科,鹅膏菌属,主要分布于我国黑龙江、吉林、四川、西藏、云南等地,有毒",
              "Boletus丽柄牛肝菌,伞菌目,牛肝菌科,牛肝菌属,分布于云南、陕西、甘肃、西藏等地,有毒",
              "Cortinarius掷丝膜菌,伞菌目,丝膜菌科,丝膜菌属,分布于湖南等地(夏秋季在山毛等阔叶林地上生长)",
              "Entoloma霍氏粉褶菌,伞菌目,粉褶菌科,粉褶菌属,主要分布于新西兰北岛和南岛西部,有毒",
              "Hygrocybe浅黄褐湿伞,伞菌目,蜡伞科,湿伞属,分布于香港(见于松仔园),有毒",
              "Lactarius松乳菇,红菇目,红菇科,乳菇属,广泛分布于亚热带松林地,无毒",
              "Russula褪色红菇,伞菌目,红菇科,红菇属,分布于河北、吉林、四川、江苏、西藏等地,无毒",
              "Suillus乳牛肝菌,牛肝菌目,乳牛肝菌科,乳牛肝菌属,分布于吉林、辽宁、山西、安徽、江西、浙江、湖南、四川、贵州等地,无毒",
              ]


def _crop_center(img, cropx, cropy):
    _, y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty + cropy, startx:startx + cropx]


def _normalize(img, mean, std):
    # This method is borrowed from:
    #   https://github.com/open-mmlab/mmcv/blob/master/mmcv/image/photometric.py
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.subtract(img, mean)
    cv2.multiply(img, stdinv)
    return img


def data_preprocess(img_path):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (256, 256))
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    img = _normalize(img.astype(np.float32), np.asarray(mean), np.asarray(std))
    img = img.transpose(2, 0, 1)
    img = _crop_center(img, 224, 224)

    return img


def resnet50_predict(args_opt):
    class_num = cfg.class_num
    local_data_path = '/cache/data'
    ckpt_file_slice = args_opt.checkpoint_path.split('/')
    ckpt_file = ckpt_file_slice[len(ckpt_file_slice)-1]
    local_ckpt_path = '/cache/'+ckpt_file

    # set graph mode and parallel mode
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

    # data download
    print('Download data.')
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=local_data_path)
    mox.file.copy_parallel(src_url=args_opt.checkpoint_path, dst_url=local_ckpt_path)

    # load checkpoint into net
    net = resnet50(class_num=class_num)
    param_dict = load_checkpoint(local_ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    # preprocess the image
    images = os.listdir(local_data_path)
    for image in images:
        img = data_preprocess(os.path.join(local_data_path, image))
        # predict model
        res = net(Tensor(img.reshape((1, 3, 224, 224)), mindspore.float32)).asnumpy()

        predict_label = label_list[res[0].argmax()]
        print("预测的蘑菇标签为:\n\t"+predict_label+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet50 train.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    parser.add_argument('--checkpoint_path', required=True, type=str, default=None, help='Checkpoint file path')
    args_opt, unknown = parser.parse_known_args()

    resnet50_predict(args_opt)
    print('ResNet50 prediction success!')
