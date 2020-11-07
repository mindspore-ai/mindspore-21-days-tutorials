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
""" Process train and eval"""
from src.config import WideDeepConfig
from mindspore import context
from src.preprocess_data import StatsDict, mkdir_path, statsdata, random_split_trans2mindrecord
from train_and_eval import train_eval
import moxing as mox


if __name__ == "__main__":
    config = WideDeepConfig()
    config.argparse_init()
    config.data_argparse_init()

    data_file = config.data_file
    data_path = config.data_path

    # data upload
    print('Upload data from obs to modelarts server.')
    mox.file.copy_parallel(src_url=config.data_url, dst_url=data_path)

    target_field_size = config.dense_dim + config.slot_dim
    stats = StatsDict(field_size=target_field_size, dense_dim=config.dense_dim, slot_dim=config.slot_dim,
                      skip_id_convert=config.skip_id_convert)
    data_file_path = data_path + data_file
    stats_output_path = data_path + "stats_dict/"
    mkdir_path(stats_output_path)
    statsdata(data_file_path, stats_output_path, stats, dense_dim=config.dense_dim, slot_dim=config.slot_dim)

    stats.load_dict(dict_path=stats_output_path, prefix="")
    stats.get_cat2id(threshold=config.threshold)

    in_file_path = data_path + data_file
    output_path = data_path + config.dataset_type
    mkdir_path(output_path)
    random_split_trans2mindrecord(in_file_path, output_path, stats, part_rows=2000000,
                                  train_line_count=config.train_line_count, line_per_sample=1000,
                                  test_size=0.1, seed=2020, dense_dim=config.dense_dim, slot_dim=config.slot_dim)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
    train_eval(config)
    print('Done all the jobs.')

