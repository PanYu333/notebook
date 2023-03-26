# 初始化

## 基础配置安装

1. 设置 `power Bluetooth` -> `off`

2. 输入命令更改apt镜像源：`sudo gedit /etc/apt/sources.list`

	- 修改deb后内容: deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/

	- 刷新镜像源：`sudo apt-get update` 

3. 安装vim：`sudo apt install vim`

4. 安装ssh服务：`sudo apt-get install openssh-server`
	- ssh服务初始化：`sudo /etc/init.d/ssh restart`



5. 网络工具：`sudo apt install net-tools`	

6. 安装gnome-tweaks桌面配置工具。(放大字体)

	- `sudo apt  install   gnome-tweaks`

	安装完成后，在终端输入下面命令，弹出优化窗口: `gnome-tweaks`

	设置font->scaling Factor-> 1.8

	

## 设置网卡静态IP

切换 su root

cd /etc/netplan

Ifconfig	获取网卡名称和IP地址

![img](file:///C:\Users\23606\AppData\Local\Temp\ksohtml15132\wps1.jpg)

route -n	获取网关，子网掩码：24 (255.255.255.0)

![img](file:///C:\Users\23606\AppData\Local\Temp\ksohtml15132\wps2.jpg)

编辑网络配置文件01-network-manager-all.yaml，内容如下：

![img](file:///C:\Users\23606\AppData\Local\Temp\ksohtml15132\wps3.jpg)





## pip修改镜像源

- 查看当前pip源： `pip config list`

- 设置清华源: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`

- 保存退出后: `pip install update`

常用镜像源：

|    阿里云    | https://mirrors.aliyun.com/pypi/simple/   |
| :----------: | ----------------------------------------- |
| 中国科技大学 | https://pypi.mirrors.ustc.edu.cn/simple/  |
|     豆瓣     | http://pypi.douban.com/simple             |
|  Python官方  | https://pypi.python.org/simple/           |
|     v2ex     | http://pypi.v2ex.com/simple/              |
|     ×××      | http://pypi.mirrors.opencas.cn/simple/    |
|   清华大学   | https://pypi.tuna.tsinghua.edu.cn/simple/ |