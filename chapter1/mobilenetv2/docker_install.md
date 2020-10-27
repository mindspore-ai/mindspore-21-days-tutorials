注：此文档内容参考MindSpore官方代码仓[README.md](https://gitee.com/mindspore/mindspore#docker%E9%95%9C%E5%83%8F)文档

### 对于GPU后端，请确保提前安装好nvidia-container-toolkit，Ubuntu用户可参考如下指南：
```
DISTRIBUTION=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$DISTRIBUTION/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit nvidia-docker2
sudo systemctl restart docker
```
然后执行如下命令，修改docker配置:
```
# 编辑daemon.json文件：
vim /etc/docker/daemon.json

# 在daemon.json里添加如下内容：
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}

# 保存并关闭daemon.json文件，然后重启docker：
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 使用以下命令拉取/启动MindSpore 1.0.0 版本GPU镜像
```
# 拉取镜像
docker pull mindspore/mindspore-gpu:1.0.0

# 启动镜像
docker run -it --runtime=nvidia --privileged=true mindspore/mindspore-gpu:1.0.0 /bin/bash
```

### 在容器终端输入`exit`命令即可退出容器
更多docker相关操作材料可自行查阅资料

