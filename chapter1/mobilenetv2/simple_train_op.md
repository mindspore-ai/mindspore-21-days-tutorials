注：需事先安装好docker gpu环境，可参考[docker_install.md](https://github.com/mindspore-ai/mindspore-21-days-tutorials/blob/main/chapter1/mobilenetv2/docker_install.md)文件

# 训练准备阶段
### 下载mobilenetv2体验脚本
```
# 在root用户主目录下执行如下命令
git clone https://github.com/mindspore-ai/mindspore-21-days-tutorials.git
mkdir -p /root/workspace/mobile
cp -r /root/mindspore-21-days-tutorials/chapter1/mobilenetv2 /root/workspace/mobile
```

### 准备cifar-10数据集（binary二进制格式）
将下载好的数据集解压，生成5个训练集.bin文件和1个测试集.bin文件。
```
# 下载并解压cifar-10数据集, 生成5个训练集.bin文件和1个测试集.bin文件
wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -zxvf cifar-10-binary.tar.gz

# 创建用于存放训练集的目录，将训练集5个.bin文件拷贝到该目录下
mkdir -p /root/workspace/mobile/data/train
cp /root/cifar-10-batches-bin/data_*.bin /root/workspace/mobile/data/train

# 创建用于存放测试集的目录，将测试集1个.bin文件拷贝到该目录下
mkdir -p /root/workspace/mobile/data/eval
cp /root/cifar-10-batches-bin/test_batch.bin /root/workspace/mobile/data/eval
```

### 训练启动阶段
##### 启动GPU容器
使用GPU mindspore-1.0.0版本镜像，将训练脚本及数据集所在目录挂载到容器环境中
```
docker run -it -v /root/workspace/mobile:/mobile --runtime=nvidia --privileged=true mindspore/mindspore-gpu:1.0.0 /bin/bash
```

##### 开始训练
```
cd /mobile/mobilenetv2
python train.py --is_training=True --epoch_size=10
```

##### 验证结果
```
python eval.py --is_training=False --pretrain_ckpt=ckpt_0/mobilenetv2-10_1562.ckpt
```

