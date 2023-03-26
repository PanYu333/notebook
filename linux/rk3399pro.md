# 初始配置

## 板端系统初始:

- 用户：`root,firefly`

- 密码：firefly

- apt更新: `sudo apt update`

 

- 网络配置工具: `apt install ethtool`





## 配置wifi静态IP：

```
network:
        wifis:
                wlan0:
                        dhcp4: false
dhcp4: false
                        addresses: [192.168.1.102/24]
                        gateway4: 192.168.1.1
                        nameservers:
                                addresses: [8.8.8.8, 114.114.114.114]
                        access-points:
                                "pc":
                                        password: "12345678"
                                "ECUST-1106-2":
                                        password: "ecust1106"

        version: 2
        renderer: NetworkManager
```





## 环境配置

### 创建python36环境

```python
sudo apt install virtualenv			# 包管理工具

#系统默认python3 -> python3.6，安装python36依赖
sudo apt-get install libpython3			
sudo apt install python3-tk

# 创建环境、激活
mkdir environment
virtualenv -p /usr/bin/python3 ~/environment/rknn_base		# 默认安装的是python36版本
source ~/environment/rknn_base/bin/activate					# 激活


# pip配置阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install update
	
conda config --show channels	# 检查是否配置成功
pip config list					# 查看当前源


# 降低一下pip版本
python -m pip install pip==20.1	# 有些东西可能安装不上

pip install -r requirement_aarch.txt	# 安装基础依赖
```



```
requirement_aarch.txt文件内容如下

numpy==1.16.3
ruamel.yaml==0.15.81
psutil==5.6.2
rknn_toolkit_lite-1.7.3-cp36-cp36m-linux_aarch64.whl
Pillow==5.3.0
torch==1.9.0
torchvision==0.10.0
torchaudio-0.10.0
scipy==1.5.4
six==1.16.0
dataclasses==0.8
pkg-resources==0.0.0
protobuf==3.19.6
typing-extensions==4.1.1
# opencv-python == 3.4.0.12 (可选)
```



- 安装其他包报错的话，降低pip和settools版本

```
python -m pip install pip==9.0.1	

python -m pip install setuptools==36.4.0
```





---

### 创建python37环境

```
# 依次安装gcc和cmake等编译环境
# 安装python3.7-tk 和 python3.7-dev
# 安装virtualenv虚拟环境
sudo apt install gcc cmake git build-essential \
python3.7-tk python3.7-dev \
virtualenv

# 创建基础环境、激活
virtualenv -p /usr/bin/python3.7m /home/firefly/environment/rknn_base_37
source /home/firefly/environment/rknn_base_37/bin/activate

```















## 更新驱动

[rknn_SDK下载](https://github.com/rockchip-linux/rknn-toolkit/releases/tag/v1.7.3)

[驱动安装包下载](https://github.com/airockchip/RK3399Pro_npu “驱动更新到1.7.3”)

- 驱动安装包中: drivers\npu_firmware\npu_fw中的所有文件替换板端的npu_fw下（find一下）





## 删除残留配置文件

1. 删除残余的配置文件
通常Debian/Ubuntu删除软件包可以用两条命令
2. remove将会删除软件包，但会保留配置文件．purge会将软件包以及配置文件都删除．

```
sudo apt-get remove <package-name>
sudo apt-get purge <package-name>
```



3. 清理缓存

	

	```python
	touch cleanBuffCache.sh		# 建立脚本
	
	内容：
	#!/bin/bash
	echo "开始清理缓存"
	# 写入硬盘，防止数据丢失
	sync;sync;sync; 
	# 延迟10S
	sleep 10
	echo 1 > /proc/sys/vm/drop_caches
	echo 2 > /proc/sys/vm/drop_caches
	echo 3 > /proc/sys/vm/drop_caches
	echo "清理缓存结束"
	```

	

给我们的定义的脚本赋予可执行的权限

```
chmod 777 cleanBuffCache.sh
```

测试一把，执行一下我们的脚本

```
./cleanBuffCache.sh
```

接下来，我们执行命令打开文件添加定时任务

```
crontab -e
```

再打开的文件中添加我们的定时任务执行的时间 以及执行的文件路径

```
* 0 * * * ./tools/clean/cleanBuffCache.sh
```

添加完成后，保存，退出

为了确保我们添加成功，可执行下面的命令查看我们是否追加成功

```
crontab -l
```