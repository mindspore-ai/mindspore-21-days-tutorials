# 训练准备阶段
### 下载mindspore 1.0.0版本代码
git clone https://gitee.com/mindspore/mindspore.git -b r1.0

### 训练及模型导出脚本准备
```
拷贝官方model_zoo提供的mobilenetv2训练脚本，做自定义修改
cp -r /root/mindspore/model_zoo/official/cv/mobilenetv2 /root/workspace/mobile/mobilenetv2

准备cifar-10数据集（二进制格式），存放于/root/workspace/mobile/data目录下，然后对脚本做一定调整
。。。

创建模型导出脚本,存放于/root/workspace/mobile/mobilenetv2目录下，命名为export_mindir.py
```

### 端侧converter_lite模型转换工具准备
##### 编译端侧converter_lite模型转换工具
进入mindspore代码仓，开始编译，converter_lite目前仅支持x86_64架构，因此-I参数值为x86_64，j8表示系统是8核CPU。
```
cd /root/mindspore
bash build.sh -I x86_64 -j8  //（约等待15分钟左右）
```
编译成功后，会在output目录下生成`mindspore-lite-1.0.0-converter-ubuntu.tar.gz`和`mindspore-lite-1.0.0-runtime-x86-cpu.tar.gz`两个压缩文件。

##### 解压并配置converter_lite工具
```
cd /root/mindspore/output
mkdir converter && mkdir runtime-x86
tar -zxvf mindspore-lite-1.0.0-converter-ubuntu.tar.gz -C ./converter --strip-components 1
tar -zxvf mindspore-lite-1.0.0-runtime-x86-cpu.tar.gz -C ./runtime-x86 --strip-components 1
```


# 训练启动阶段
### 启动GPU容器
使用GPU mindspore-1.0.0版本镜像，将端侧编译工具所在目录及训练脚本所在目录挂载到容器环境中
```
docker run -it -v /root/mindspore/output:/mslite_lib/ -v /root/workspace/mobile:/mobile --runtime=nvidia --privileged=true mindspore/mindspore-gpu:1.0.0 /bin/bash
```

### 配置converter_lite所需库环境
```
cp /mslite_lib/converter/converter/converter_lite /usr/local/bin
export LD_LIBRARY_PATH=/mslite_lib/converter/third_party/protobuf/lib:/mslite_lib/converter/third_party/flatbuffers/lib:/mslite_lib/runtime-x86/lib:${LD_LIBRARY_PATH}
```

### 开始训练
```
cd /mobile/mobilenetv2
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

# 操作步骤视频
[视频访问地址](https://mslite-app.obs.cn-north-4.myhuaweicloud.com:443/%E6%93%8D%E4%BD%9C%E8%A7%86%E9%A2%91-%E7%BB%88%E7%89%882.mp4?AccessKeyId=PQ7DQUATQUMX3VMMPIPM&Expires=1603523256&Signature=wHXWu5LQd%2BFcKXfygZDbROvuLzY%3D)


