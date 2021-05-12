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
import os
import tarfile


def count_line(filepath):
    count = 0
    f = open(filepath, "r")
    for line in f.readlines():
        count = count + 1
    return count


def find_ckpt(ckpt_path):
    files = os.listdir(ckpt_path)
    for fi in files:
        fi_d = os.path.join(ckpt_path, fi)
        if fi.endswith(".ckpt"):
          return fi_d


def untar(fname, dirs):
    try:
        t = tarfile.open(fname)
        t.extractall(path = dirs)
        return True
    except Exception as e:
        print(e)
        return False
