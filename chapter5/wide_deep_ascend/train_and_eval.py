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

""" train and eval """

import os
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net,\
    build_searched_strategy, merge_sliced_parameter
from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.util import find_ckpt
import moxing as mox


def get_WideDeep_net(config):
    """
    Get network of wide&deep model.
    """
    WideDeep_net = WideDeepModel(config)

    loss_net = NetWithLossClass(WideDeep_net, config)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(WideDeep_net)

    return train_net, eval_net


class ModelBuilder():
    """
    Wide and deep model builder
    """
    def __init__(self):
        pass

    def get_hook(self):
        pass

    def get_train_hook(self):
        hooks = []
        callback = LossCallBack()
        hooks.append(callback)

        if int(os.getenv('DEVICE_ID')) == 0:
            pass
        return hooks

    def get_net(self, config):
        return get_WideDeep_net(config)


def train_eval(config):
    """
    test evaluate
    """
    data_path = config.data_path + config.dataset_type
    ckpt_path = config.ckpt_path
    epochs = config.epochs
    batch_size = config.batch_size
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5

    ds_train = create_dataset(data_path, train_mode=True, epochs=1,
                              batch_size=batch_size, data_type=dataset_type)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))
    ds_eval = create_dataset(data_path, train_mode=False, epochs=1,
                             batch_size=batch_size, data_type=dataset_type)
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    net_builder = ModelBuilder()
    train_net, eval_net = net_builder.get_net(config)
    train_net.set_train()

    train_model = Model(train_net)
    train_callback = LossCallBack(config=config)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(),
                                  keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train', directory=config.ckpt_path, config=ckptconfig)
    train_model.train(epochs, ds_train, callbacks=[TimeMonitor(ds_train.get_dataset_size()), train_callback, ckpoint_cb])

    # data download
    print('Download data from modelarts server to obs.')
    mox.file.copy_parallel(src_url=config.ckpt_path, dst_url=config.train_url)

    param_dict = load_checkpoint(find_ckpt(ckpt_path))
    load_param_into_net(eval_net, param_dict)

    auc_metric = AUCMetric()
    eval_model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})
    eval_callback = EvalCallBack(eval_model, ds_eval, auc_metric, config)

    eval_model.eval(ds_eval, callbacks=eval_callback)

