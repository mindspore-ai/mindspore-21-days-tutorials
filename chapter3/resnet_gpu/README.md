# MindSpore ResNet-50毒蘑菇识别教程指导（GPU环境）

该教程旨在指导大家使用GPU资源完成MindSpore ResNet-50毒蘑菇识别的教程。

> **注意：** 该教程的代码是基于`v1.0`版本的MindSpore [ModelZoo](https://github.com/mindspore-ai/mindspore/tree/r1.0/model_zoo/official/cv/resnet)开发完成的。

## 上手指导

### 安装系统库

* 系统库

    ```
    sudo apt install -y unzip
    ```

* Python库

    ```
    pip install opencv-python
    ```

* MindSpore (**v1.0**)

    MindSpore的安装教程请移步至 [MindSpore安装页面](https://www.mindspore.cn/install).

### 下载蘑菇数据集

```
cd mushroom-dataset/ && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/mushrooms/mushrooms.zip
unzip mushrooms.zip && rm mushrooms.zip
cd ../resnet_gpu/
```

或者您可以直接点击 [https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/mushrooms/mushrooms.zip](https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/mushrooms/mushrooms.zip) 从浏览器中下载该数据集，手动解压。

### 模型训练

```
python train.py --dataset_path ../mushroom-dataset/train
```
```
epoch: 90 step: 201, loss is 1.2514226
epoch: 90 step: 202, loss is 1.033073
epoch: 90 step: 203, loss is 1.3107909
epoch: 90 step: 204, loss is 1.0333312
epoch: 90 step: 205, loss is 1.255686
epoch: 90 step: 206, loss is 1.1876906
epoch: 90 step: 207, loss is 1.123241
epoch: 90 step: 208, loss is 1.2708801
epoch: 90 step: 209, loss is 1.1666311
Epoch time: 11931.645, per step time: 57.089
```

### 下载ResNet-50预训练模型（推理任务使用）

```
cd ./ckpt_files && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/ckpt_files/resnet-90_209.ckpt
```

或者您可以直接点击 [https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/ckpt_files/resnet-90_209.ckpt](https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/resnet-50/ckpt_files/resnet-90_209.ckpt) 从浏览器中下载预训练模型。

### 模型精度验证

```
python eval.py --checkpoint_path ./ckpt_files/resnet-90_209.ckpt --dataset_path ../mushroom-dataset/eval
```
```
result: {'top_5_accuracy': 0.9895833333333334, 'top_1_accuracy': 0.78125} ckpt= resnet-90_209.ckpt
```

### 模型推理

```
python predict.py --checkpoint_path ./ckpt_files/resnet-90_209.ckpt --image_path ./tum.jpg
```
```
预测的蘑菇标签为:
	Agaricus双孢蘑菇,伞菌目,蘑菇科,蘑菇属,广泛分布于北半球温带,无毒
```

## 许可证

[Apache License 2.0](../../LICENSE)
