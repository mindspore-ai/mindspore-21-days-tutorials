# MindSpore YOLOv3-DarkNet53篮球检测教程指导（GPU环境）

该教程旨在指导大家使用GPU资源完成MindSpore YOLOv3-DarkNet53篮球检测的教程。

> **注意：** 该教程的代码是基于`v1.0`版本的MindSpore [ModelZoo](https://github.com/mindspore-ai/mindspore/tree/r1.0/model_zoo/official/cv/yolov3_darknet53)开发完成的。

> **注意：** 考虑到预训练过程会占用大量时间，本次课程我们不会提供完整的数据集用于模型训练，但我们会提供YOLOv3预训练模型以及测试数据集，方便大家用于模型验证和推理工作。

## 上手指导

### 安装系统库

* 系统库

    ```
    sudo apt install -y unzip
    ```

* Python库

    ```
    pip install opencv-python pycocotools
    ```

* MindSpore (**v1.0**)

    MindSpore的安装教程请移步至 [MindSpore安装页面](https://www.mindspore.cn/install).

### 数据准备

* 下载测试数据集（验证任务使用）

    ```
    cd basketball-dataset/ && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/yolov3-darknet53/basketball-dataset/basketball-dataset.zip
    unzip basketball-dataset.zip && rm basketball-dataset.zip
    ```

    或者您可以直接点击 [https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/yolov3_darknet53/basketball-dataset/basketball-dataset.zip](https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/yolov3_darknet53/basketball-dataset/basketball-dataset.zip) 从浏览器中下载该数据集，手动解压。

* 下载YOLOv3-DarkNet53预训练模型（验证/推理任务使用）

    ```
    cd ../resnet_gpu/ckpt_files && wget https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/yolov3_darknet53/ckpt_files/yolov3-320_168000.ckpt
    ```

    或者您可以直接点击 [https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/yolov3_darknet53/ckpt_files/yolov3-320_168000.ckpt](https://ascend-tutorials.obs.cn-north-4.myhuaweicloud.com/yolov3_darknet53/ckpt_files/yolov3-320_168000.ckpt) 从浏览器中下载预训练模型。

### 模型验证

```
python eval.py --data_dir ../basketball-dataset/ --pretrained ./ckpt_files/yolov3-320_168000.ckpt
```
```
=============coco eval result=========
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.568
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.829
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.716
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.550
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.568
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.645
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.646
```

### 模型推理

```
python predict.py --image_path ./00086.jpg --pretrained ./ckpt_files/yolov3-320_168000.ckpt
```

输入图片：

<img src="../docs/00086.jpg" alt="Input Image" width="600"/>

输出结果：

<img src="../docs/output.jpg" alt="Output Image" width="600"/>

## 许可证

[Apache License 2.0](../../LICENSE)
