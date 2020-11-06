注：需事先安装好docker gpu环境，可参考[docker_install.md](https://github.com/mindspore-ai/mindspore-21-days-tutorials/blob/main/chapter1/mobilenetv2/docker_install.md)文件

# 训练准备阶段
### 下载wide & deep体验脚本
```
# 在root用户主目录下执行如下命令
git clone https://github.com/mindspore-ai/mindspore-21-days-tutorials.git
mkdir -p /root/workspace/wide_deep
cp -r /root/mindspore-21-days-tutorials/chapter5/wide_deep /root/workspace/wide_deep
cd /root/workspace/wide_deep
```

### 准备阶段
##### 体验作业准备
下载事先准备好的mindrecord和.ckpt文件
```
# 从华为云obs上下载经由10%Criteo数据集训练生成的mindrecord数据集文件
wget https://wide-deep-21.obs.cn-north-4.myhuaweicloud.com/train_demo.tar.gz
tar -zxvf train_demo.tar.gz
mkdir -p data/ten_percent
mv mindrecord data/ten_percent
# 从华为云obs上下载经由10%Criteo数据集训练生成的.ckpt文件
wget https://wide-deep-21.obs.cn-north-4.myhuaweicloud.com/wide_deep.ckpt
```

##### 进阶作业准备
准备Criteo数据集（非全量），从华为云obs上下载Criteo数据集mini_demo.txt（全量数据的1%）
```
mkdir -p data/one_percent
wget https://wide-deep-21.obs.cn-north-4.myhuaweicloud.com/mini_demo.txt
mv mini_demo.txt ./data/one_percent
```


### 训练启动阶段
##### 启动GPU容器
使用GPU mindspore-1.0.0版本镜像，将训练脚本及数据集所在目录挂载到容器环境中
```
docker run -it -v /root/workspace/wide_deep:/wide_deep --runtime=nvidia --privileged=true mindspore/mindspore-gpu:1.0.0 /bin/bash
```

##### 安装环境依赖项
```
pip install pandas
pip install sklearn
```
若执行过程中出现如下警告，可执行`pip install --upgrade pip`命令升级工具.
```
WARNING: You are using pip version 19.2.3, however version 20.2.4 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
```

##### 体验作业要求
验证结果
```
cd /wide_deep
python eval.py --data_path=data/ten_percent/mindrecord --ckpt_path=wide_deep.ckpt
```

##### 进阶作业要求
处理数据
```
python src/preprocess_data.py --data_path=/wide_deep/data/one_percent/ --data_file=mini_demo.txt
```

开始训练
```
python train.py --data_path=data/one_percent/mindrecord
```

验证结果
```
python eval.py --data_path=data/one_percent/mindrecord --ckpt_path=widedeep_train-1_42.ckpt
```



