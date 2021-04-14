注：需事先安装好docker gpu环境，可参考[docker_install.md](https://github.com/mindspore-ai/mindspore-21-days-tutorials/blob/main/chapter1/mobilenetv2/docker_install.md)文件。

# 训练准备阶段
### 启动GPU容器
使用GPU mindspore-1.1.1版本镜像，启动容器
```
docker run -it --runtime=nvidia --privileged=true mindspore/mindspore-gpu:1.1.1 /bin/bash
```

### 下载mindspore 1.1.1版本代码
```
# 在容器root用户主目录执行如下命令
git clone https://gitee.com/mindspore/mindspore.git -b r1.1
```

### 训练及模型导出脚本准备
##### 拷贝官方model_zoo提供的mobilenetv2训练脚本，做自定义修改
```
mkdir -p /root/workspace/mobile
cp -r /root/mindspore/model_zoo/official/cv/mobilenetv2 /root/workspace/mobile/mobilenetv2
```
若想快速体验运行脚本，并不想做自定义修改，可不执行上述cp命令,直接使用如下命令下载第一讲课程提供的[mobilenetv2脚本](https://github.com/mindspore-ai/mindspore-21-days-tutorials/tree/main/chapter1/mobilenetv2)，将mobilenetv2目录拷贝至/root/workspace/mobile目录下即可。

```
git clone https://github.com/mindspore-ai/mindspore-21-days-tutorials.git
cp -r /root/mindspore-21-days-tutorials/chapter1/mobilenetv2 /root/workspace/mobile
```

##### 准备cifar-10数据集（binary二进制格式）
使用tar命令将下载好的数据集解压，生成5个训练集.bin文件和1个测试集.bin文件。
```
# 创建用于存放训练集的目录，将训练集5个.bin文件拷贝到该目录下
mkdir -p /root/workspace/mobile/data/train

# 创建用于存放测试集的目录，将测试集1个.bin文件拷贝到该目录下
mkdir -p /root/workspace/mobile/data/eval
```

### 端侧converter_lite模型转换工具准备】
##### 编译端侧converter_lite模型转换工具
进入mindspore代码仓，开始编译，converter_lite目前仅支持x86_64架构，因此-I参数值为x86_64，j8表示系统是8核CPU。
```
cd /root/mindspore
bash build.sh -I x86_64 -j8  //（约等待15分钟左右）
```
编译成功后，会在output目录下生成`mindspore-lite-1.1.1-converter-linux-64.tar.gz`和`mindspore-lite-1.1.1-inference-linux-x64.tar.gz`

##### 解压并配置converter_lite工具
```
cd /root/mindspore/output
mkdir -p /usr/local/converter
tar -zxvf mindspore-lite-1.1.1-converter-linux-64.tar.gz -C /usr/local/converter --strip-components 1
```

##### 配置converter_lite所需库环境
```
export LD_LIBRARY_PATH=/usr/local/converter/lib:/usr/local/converter/third_party/glog/lib:${LD_LIBRARY_PATH}
```

# 训练启动阶段
### 开始训练
```
cd /root/workspace/mobile/mobilenetv2
python train.py --is_training=True --epoch_size=10
```

### 验证结果
```
python eval.py --is_training=False --pretrain_ckpt=ckpt_0/mobilenetv2-10_1562.ckpt
```

### 导出.mindir模型文件
```
python export_mindir.py --ckpt_path=ckpt_0/mobilenetv2-10_1562.ckpt
```

### 转换生成端侧模型
```
converter_lite --fmk=MINDIR --modelFile=./mobilenetv2.mindir --outputFile=mobilenetv2
```
