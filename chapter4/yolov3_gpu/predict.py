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
"""YoloV3 eval."""
import os
import argparse
import datetime
import sys
from collections import defaultdict

import cv2
import numpy as np

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.yolo import YOLOV3DarkNet53
from src.logger import get_logger
from src.config import ConfigYOLOV3DarkNet53
from src.transforms import _reshape_data

label_list = ['stand', 'walk', 'run', 'shoot', 'defense']


class DetectionEngine:
    """Detection engine."""

    def __init__(self, args):
        self.ignore_threshold = args.ignore_threshold
        self.labels = label_list
        self.num_classes = len(self.labels)
        self.results = defaultdict(list)
        self.det_boxes = []
        self.nms_thresh = args.nms_thresh

    def do_nms_for_results(self):
        """Get result boxes."""
        for clsi in self.results:
            dets = self.results[clsi]
            dets = np.array(dets)
            keep_index = self._nms(dets, self.nms_thresh)

            keep_box = [{'category_id': self.labels[int(clsi)],
                         'bbox': list(dets[i][:4].astype(float)),
                         'score': dets[i][4].astype(float)}
                        for i in keep_index]
            self.det_boxes.extend(keep_box)

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indexs = np.where(ovr <= threshold)[0]
            order = order[indexs + 1]
        return reserved_boxes

    def detect(self, outputs, batch, image_shape, config=None):
        """Detect boxes."""
        outputs_num = len(outputs)
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_id in range(outputs_num):
                # 32, 52, 52, 3, 85
                out_item = outputs[out_id]
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                # get number of items in one head, [B, gx, gy, anchors, 5+80]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]

                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                cls_emb = cls_emb.reshape(-1, config.num_classes)
                conf = conf.reshape(-1)
                cls_argmax = cls_argmax.reshape(-1)

                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                # create all False
                flag = np.random.random(cls_emb.shape) > sys.maxsize
                for i in range(flag.shape[0]):
                    c = cls_argmax[i]
                    flag[i, c] = True
                confidence = cls_emb[flag] * conf
                for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence, cls_argmax):
                    if confi < self.ignore_threshold:
                        continue
                    x_lefti = max(0, x_lefti)
                    y_lefti = max(0, y_lefti)
                    wi = min(wi, ori_w)
                    hi = min(hi, ori_h)
                    # transform catId to match coco
                    coco_clsi = str(clsi)
                    self.results[coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])

    def draw_boxes_in_image(self, img_path):
        img = cv2.imread(img_path, 1)
        for i in range(len(self.det_boxes)):
            x = int(self.det_boxes[i]['bbox'][0])
            y = int(self.det_boxes[i]['bbox'][1])
            w = int(self.det_boxes[i]['bbox'][2])
            h = int(self.det_boxes[i]['bbox'][3])
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 225, 0), 1)
            score = round(self.det_boxes[i]['score'], 3)
            text = self.det_boxes[i]['category_id']+', '+str(score)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

        return img


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser('mindspore coco testing')
    # device related
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: GPU)')
    # dataset related
    parser.add_argument('--image_path', required=True, type=str, default='', help='image file path')
    parser.add_argument('--output_dir', type=str, default='./', help='image file output folder')
    parser.add_argument('--per_batch_size', default=1, type=int, help='batch size for per gpu')
    # network related
    parser.add_argument('--pretrained', required=True, default='', type=str,
                        help='model_path, local pretrained model to load')
    # logging related
    parser.add_argument('--log_path', type=str, default='outputs/', help='checkpoint save location')
    # detect_related
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='threshold for NMS')
    parser.add_argument('--ignore_threshold', type=float, default=0.01,
                        help='threshold to throw low quality boxes')

    args, _ = parser.parse_known_args()
    return args


def data_preprocess(img_path, config):
    img = cv2.imread(img_path, 1)
    img, ori_image_shape = _reshape_data(img, config.test_img_shape)
    img = img.transpose(2, 0, 1)

    return img, ori_image_shape


def predict():
    """The function of predict."""
    args = parse_args()

    devid = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=False, device_id=devid)

    # logger
    args.outputs_dir = os.path.join(args.log_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    rank_id = int(os.environ.get('RANK_ID')) if os.environ.get('RANK_ID') else 0
    args.logger = get_logger(args.outputs_dir, rank_id)

    args.logger.info('Creating Network....')
    network = YOLOV3DarkNet53(is_training=False)

    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.pretrained))
    else:
        args.logger.info('{} not exists or not a pre-trained file'.format(args.pretrained))
        assert FileNotFoundError('{} not exists or not a pre-trained file'.format(args.pretrained))
        exit(1)

    config = ConfigYOLOV3DarkNet53()
    args.logger.info('testing shape: {}'.format(config.test_img_shape))
    # data preprocess operation
    image, image_shape = data_preprocess(args.image_path, config)

    # init detection engine
    detection = DetectionEngine(args)

    input_shape = Tensor(tuple(config.test_img_shape), ms.float32)
    args.logger.info('Start inference....')
    network.set_train(False)
    prediction = network(Tensor(image.reshape(1, 3, 416, 416), ms.float32), input_shape)
    output_big, output_me, output_small = prediction
    output_big = output_big.asnumpy()
    output_me = output_me.asnumpy()
    output_small = output_small.asnumpy()

    detection.detect([output_small, output_me, output_big], args.per_batch_size,
                     image_shape, config)
    detection.do_nms_for_results()
    img = detection.draw_boxes_in_image(args.image_path)

    cv2.imwrite(os.path.join(args.output_dir, 'output.jpg'), img)


if __name__ == "__main__":
    predict()
