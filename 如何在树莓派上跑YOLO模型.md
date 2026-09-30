# 如何在树莓派上跑YOLO模型

## 如何在树莓派上跑YOLO模型

############################



<div align="center" style="border: 3px solid gray;border-radius: 27px;overflow: hidden;"> <a class="link-info" href="https://player.bilibili.com/player.html?aid=115122169191399&autoplay=0" rel="nofollow" title="【玩转硬件】手把手教你在树莓派上运行YOLO模型">【玩转硬件】手把手教你在树莓派上运行YOLO模型</a><iframe id="sfOIaOdm-1758098571100" frameborder="0" src="https://player.bilibili.com/player.html?aid=115122169191399&amp;autoplay=0" allowfullscreen="true" data-mediaembed="bilibili" style="width: 100%; aspect-ratio: 2;" allow="fullscreen" loading="lazy"></iframe></div>



############################

目前，AI已经走进了各家各户，众所周知，AI是需要算力的，如果缺乏算力，目前的各种大模型之类的都是无法很好的或者说很流畅的运行起来。在工业界或者是在一些特殊的场景中，控制功耗是十分重要的，如果只是由于需要AI而增加部件的部署成本或者是散热成本的话对部件整体的设计影响比较大。所以，如何高效地在轻量化的硬件上面运行AI模型也是一个非常有效的课题。我们今天开一个新的系列，新系列的名称为玩板子！主要是依靠树莓派、香橙派等开发板为基础，开发一系列的AI应用。第一期视频我们做的会比较基础一些，今天我们将会以树莓派为例，完成一个yolo模型的构建，并且在树莓派上构建一个图形化的系统，利用树莓派完成图片的检测和视频的检测。

首先，我们还是需要看一下你会做成一个什么样子的内容，如下图所示，首先第一张图是我们的硬件图，我们有一个树莓派的板子，并且这个板子上面添加了一个树莓派的（Camera Serial Interface）摄像头。树莓派的hdmi接口插在一个显示器上，这里有我们制作的图形化的系统，在这个系统中，你可以使用树莓派去做YOLO模型的加载、利用模型进行图像的预测或者是利用模型去做视频的预测。

 ![image-20250830191440040](https://i-blog.csdnimg.cn/img_convert/ae65add0bc69749f32a8496f2737221d.png)

 ![image-20250830191420575](https://i-blog.csdnimg.cn/img_convert/315ddd1340466c748fb62679e195d3d3.png)

### 前期准备

#### 啥是树莓派

树莓派（Raspberry Pi）是一款由英国树莓派基金会开发的低成本、信用卡大小的单板计算机，旨在促进计算机科学教育。它具备基本的计算功能，可用于编程、电子项目、媒体中心等多种用途。

 ![将电源插入树莓派](https://i-blog.csdnimg.cn/img_convert/37bca5cd39e663308b24fa72753e1100.png)

树莓派整体上就是这样的一块板子，为了方便理解，你可以将理解为一块完整的电脑主板，麻雀虽小，五脏俱全。在这个小小的板子上面，有CPU、有硬盘（SD卡）、有内存以及配备HDMI、USB、GPIO等接口，支持多种外设连接。如果只是浏览一下网页，看一下视频而已，这个板子完全可以作为你的小电脑主机。总的来说，树莓派是一款功能强大、价格低廉的单板计算机，广泛应用于教育、物联网、媒体中心等领域，适合初学者和开发者使用。

当然，他们在性能上面和我们家用电脑肯定是相差很大的。家用的电脑普遍使用Intel、AMD等CPU，采用的是X86的架构，性能比较强，适用桌面和服务器的应用。对于树莓派而言，他使用的是ARM的架构，功耗比较低，适合各种嵌入式系统和移动设备。INTEL和AMD的CPU功耗通常在65W到150W之间，对于树莓派而言，他的功耗只有几w。在操作系统方面，INTEL和AMD支持我们常见的windows、linux以及macos等多种操作系统，对于树莓派而言，他主要支持的是一些开源的Raspberry Pi OS、Linux等系统。另外，在扩展性方面，树莓派可以GPIO接口进行扩展，可以连接一些传感器或者是舵机等设备，对于Intel/AMD而言，他们主要是通过PCIe、USB等接口扩展，适合高性能外设。

 ![image-20250206161626927](https://i-blog.csdnimg.cn/img_convert/55a483f0eac6f6d209d930dd4ff894a3.png)

#### 树莓派的基础使用常识

对于更多的树莓派的基础知识，大家可以查看树莓派的官方文档。

这里是官方文档的地址： [入门 | Raspberry Pi 树莓派 (官网25年2月更新)](https://pidoc.cn/docs/computers/getting-started/) 

 ![8955167b769aa07e51f550c24455d1f](https://i-blog.csdnimg.cn/img_convert/f4a9befd55753654619fbd215bde6d35.jpeg)

### 在树莓派上运行YOLO模型

我们今天的目的主要是希望在一个新的树莓派板子上成功运行yolo模型。那首先你要有一个板子，以我自己为例，我将我提前准备好的内容的列表放在下面。

- 树莓派主板（具体型号: 树莓派4B 8GB）

- 支持HDMI协议的显示器（可以是家用的显示器）

- 64GB的TF卡和读卡器

- 散热片

- micro HDMI线

- 网线（也可以直接通过WIFI进行联网）

- typec开关的电源

- 亚克力的外壳

- 500w像素的广角摄像头

 ![image-20250206162659016](https://i-blog.csdnimg.cn/img_convert/a2c638fd06e715567ddaeb57a8c92dbd.png)

准备好这些内容之后，首先将这些设备安装起来，尤其是将摄像头插好，这里的摄像头不是USB的摄像头，是CSI的摄像头，插的时候需要注意接口，如下图所示。

 ![img](https://i-blog.csdnimg.cn/img_convert/726d719e0a5e6bb6b5a26a958f1ec956.jpeg)

下面就是我直接组装好的树莓派的整体情况了。（PS: 本人动手能力较差，组装半天才安排明白…）

 ![7c260869cc0aeb07ab4ebcd74d76c96](https://i-blog.csdnimg.cn/img_convert/a91c312abd92b8b357c96c24aba98d88.jpeg)

#### 邮件配置（可选）

因为我们的系统有一个发送邮件的功能，发送邮件是通过邮箱的smtp服务来支持的，所以需要提前配置一下你邮箱的smtp服务，我这里以qq邮箱为例进行演示。

首先登录你的qq邮箱，找到对应的设置部分。

 ![image-20250830151611102](https://i-blog.csdnimg.cn/img_convert/24a6ff20d48d22d5ed6c71fa7a257b7a.png)

找到下方的服务部分，如果没开启的需要你开启这个服务，然后完成后面的手机验证部分。

 ![image-20250830151740448](https://i-blog.csdnimg.cn/img_convert/e5d79336ee99fc661f2f4b19b3ffaab5.png)

绑定你对应的手机，点击生成授权码，将会出现一串数字，你需要将这串数字保存，后面在系统配置的时候将会使用到。

 ![image-20250830152128571](https://i-blog.csdnimg.cn/img_convert/22b25a2cf73bed7f3b83b24e147a588a.png)

 ![image-20250830152201988](https://i-blog.csdnimg.cn/img_convert/03a5a68420ed36a85835487e527180a5.png)

在代码中填写下面的信息

 ![image-20250830153405132](https://i-blog.csdnimg.cn/img_convert/9e60d1f7fed800e6f99eaf017a6daf9a.png)

测试成功之后就是这个样子的。

 ![image-20250830153308968](https://i-blog.csdnimg.cn/img_convert/e172a6e755dfbce663b20738b6f51e33.png)

#### 资源获取

请看这里：https://space.bilibili.com/161240964

#### 烧录系统

接下来就是拿着我们的内存卡来进行系统镜像的烧录了。（注意：系统镜像的烧录指的是把你的内存卡通过读卡器插在你自己的windows电脑上进行操作。）

镜像的烧录我们直接使用官方的烧录软件进行即可，这个软件的下载地址是： [Raspberry Pi OS – Raspberry Pi](https://www.raspberrypi.com/software/) 

历史镜像的下载地址： [Index of /raspios_arm64/images](https://downloads.raspberrypi.com/raspios_arm64/images/) 

为了保险起见，防止后面的好朋友看的时候时间已经过去很长时间，为了保证你可以按照教程复现出来，这里我使用的是这个版本。

https://downloads.raspberrypi.com/raspios_arm64/images/raspios_arm64-2024-11-19/2024-11-19-raspios-bookworm-arm64.img.xz

 ![image-20250403170546604](https://i-blog.csdnimg.cn/img_convert/3e7ffe1fb63af8c618007e9cde9c7a2b.png)

软件安装启动之后长这个样子，第一个按钮是用来选择树莓派板子的型号的，第二个是选择操作系统的， 第三个则是用来选择你要进行烧录的内存卡的，这个内存卡后面将会类似于硬盘一样插在你的树莓派的板子上面。

 ![image-20250403170624516](https://i-blog.csdnimg.cn/img_convert/c312a548279fddcc1b6cdc176e16a941.png)

需要注意的是！！！你的操作系统这里的选择，如果你想要避免一些不必要的麻烦，请你务必和主播的版本选择一致，这样就可以减少后面版本不一致导致的各种库的安装问题。我们在这里选择的是64位的系统，64位系统占用的资源会更多，响应的能力也会更强，32位的系统虽然占用的资源少，但是遇到的麻烦会更多，尤其是在找对应的库的时候要踩的坑超级多，本次我们就来使用这个官方的树莓派64位系统！

 ![image-20250403171202859](https://i-blog.csdnimg.cn/img_convert/87292ee1ce6d2835872176d9152155cf.png)

为了防止后期官方更新之后找不到对应的镜像文件，我也会在我的资源包中放置这个镜像文件，方便大家来使用。

等待片刻，等待烧录完成即可，之后你就可以取下你的读卡器，然后将你的sd卡从你的读卡器中取出。

烧录的过程中记得对密码以及wifi密码进行配置，方便后面登录和远程连接。

 ![image-20250817234700638](https://i-blog.csdnimg.cn/img_convert/252da1ccae02a36d3909762cdc3ee26b.png)

第二个选项卡里面的ssh一定要打开，打开之后方便我们使用远程的软件进行登录。

 ![image-20250817234739925](https://i-blog.csdnimg.cn/img_convert/80f48b87b799247cfa364777416e2099.png)

选择完毕之后直接烧就可以了。

 ![image-20250817234827373](https://i-blog.csdnimg.cn/img_convert/6f86f384251d8204262afba3c12d825a.png)

点击确认然后点击开始。

 ![image-20250817234947494](https://i-blog.csdnimg.cn/img_convert/f5d9a9c18367acd56a5883ac8081edd8.png)

写入成功之后在软件界面上会提示，然后直接取下你的读卡器就可以了。

 ![image-20250818000349054](https://i-blog.csdnimg.cn/img_convert/d4b70546fefbef955d1255ed44c32f69.png)

#### 树莓派启动！

##### 查找IP地址并连接你的电脑

树莓派启动之后，如果你有显示器的话您可以直接使用随树莓派携带的micro hdmi线连接显示器然后直接启动树莓派即可。

 ![image-20250830144858870](https://i-blog.csdnimg.cn/img_convert/9da7610f40b5e0eff5ee293df02cc7e2.png)

首先，我们进入命令行，通过下面的指令查看ip地址。

```bash
hostname -i
ifconfig
```

如果没有显示器的话，请确保您已经按照上面的方式正确配置了wifi账号和密码，当设备接电启动之后，在手机上下载一个fing的软件，并对整体的网络进行扫描即可发现你设备的ip地址，如下是我扫描到我树莓派的ip地址。

 ![image-20250818004217791](https://i-blog.csdnimg.cn/img_convert/33f9134acbfa5d52ddd90fd08ae56168.png)

之后我们需要下载mobaxterm： [MobaXterm Xserver with SSH, telnet, RDP, VNC and X11 - Home Edition](https://mobaxterm.mobatek.net/download-home-edition.html) 

这个软件将会帮助我们后面连接我们的树莓派以及向我们的树莓派中上传文件。

下载完成之后新建一个会话，在下方的位置中输入刚才在fing软件中扫描到的ip地址。

 ![image-20250830144745685](https://i-blog.csdnimg.cn/img_convert/b40c3068a42adad0b40708f11716aeb7.png)

然后点击连接即可。

 ![image-20250818004029058](https://i-blog.csdnimg.cn/img_convert/f994a8fbefefe78153d6538bf33c8377.png)

输入账号密码之后即可连接成功。

 ![image-20250818004151024](https://i-blog.csdnimg.cn/img_convert/c468daccb8bf1eb32f524f4bb3634e2e.png)

这是其中一种通过命令行的方式连接我们的树莓派。

当然为了我们能够直接看到画面，我们这里可以对我们的树莓派进行一些设置，通过VNC的方式来访问我们的树莓派。

VNC的下载地址在这里： [Download VNC Viewer by RealVNC®](https://www.realvnc.com/en/connect/download/viewer/) 

您需要在你的笔记本或者台式机上下载VNC VIEWER并进行安装。

在终端输入以下命令进入配置界面。

```
sudo raspi-config
```

输入之后出现下面的界面

 ![image-20250818004850717](https://i-blog.csdnimg.cn/img_convert/228f9741e2d897c9113870799c3fde73.png)

依次操作：Interfacing Options -> VNC -> Yes。之后系统会提示你是否要安装 VNC 服务，输入 y 之后回车，等待系统自动下载安装完成，一切顺利的话 VNC 服务就启动了！

 ![image-20250818004924990](https://i-blog.csdnimg.cn/img_convert/1a7bac937734901630b86b9b1360fc0f.png)

稍等片刻，

 ![image-20250818004956893](https://i-blog.csdnimg.cn/img_convert/dc657313ffb302a3e74f78c46a7922b7.png)

然后我们重启一下我们的树莓派，输入下列指令重启我们的树莓派。

```
sudo reboot
```

在vnc中我们输入树莓派的ip地址。

 ![image-20250818005431490](https://i-blog.csdnimg.cn/img_convert/16622bcbe4b62e241dfdff548376d877.png)

连接我们的树莓派。

 ![image-20250818005302911](https://i-blog.csdnimg.cn/img_convert/06835c2f4096fe398e7cd9384ef5a878.png)

输入我们刚才的账号密码。

 ![image-20250818005329605](https://i-blog.csdnimg.cn/img_convert/c4c61cf42ccc0948c988a02c247ef59f.png)

这个时候出现了我们的图形化界面，完美。

 ![image-20250818005354041](https://i-blog.csdnimg.cn/img_convert/37ab744887c4514855923c9c5ad2a374.png)

下面我们需要对软件进行一次更新，首先为了确保大家和我这里的树莓派系统一致，我将我的系统版本信息的截图放在下面。

 ![image-20250818005814583](https://i-blog.csdnimg.cn/img_convert/dd9a21cf5b5dad82a3ad44126a41860d.png)

 ![image-20250818010247262](https://i-blog.csdnimg.cn/img_convert/2d278176daa755c70bbe3871411ce417.png)

源的切换我们这里参考树莓派实验室： [Raspberry Pi OS 中国软件源 | 树莓派实验室](https://shumeipai.nxez.com/2024/11/11/raspberry-pi-os-chinese-software-source.html) 

备份原始文件（可选步骤）

```bash
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo cp /etc/apt/sources.list.d/raspi.list /etc/apt/sources.list.d/raspi.list.bak
```

编辑软件源配置

1、编辑 /etc/apt/sources.list 文件（软件源），参考如下命令：

```
sudo chmod 777 /etc/apt/sources.list
sudo nano /etc/apt/sources.list
```

2、删除原文件所有内容， **bookworm** 系统用以下内容取代：

```
deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free-firmware
deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free-firmware
```

 ![image-20250818010935299](https://i-blog.csdnimg.cn/img_convert/cb023945722ea8e5e857eefb20a5589c.png)

Ctrl+o 保存，之后回车确认，然后 Ctrl+x 退出。

编辑系统源配置

1、编辑 /etc/apt/sources.list.d/raspi.list 文件（系统更新源），参考如下命令：

```
sudo chmod 777 /etc/apt/sources.list.d/raspi.list
sudo nano /etc/apt/sources.list.d/raspi.list
```

2、同样修改首行网址，修改后文件如下：

```
deb https://mirrors.tuna.tsinghua.edu.cn/raspberrypi bookworm main
```

bookworm、bullseye、buster、stretch、jessie、wheezy 版本的系统按照之前修改软件源的的规则修改即可，这里不再赘述。
Ctrl+o 保存，之后回车确认，然后 Ctrl+x 退出。

配置好了可以尝试更新，用下面的命令分别更新软件源列表、软件版本和系统内核版本，完整的更新过程需要等挺久的。一般只用更新软件源列表即可。

如果更新过程总提示出现了下面的进程锁定的错误，可以按照下面的截图杀死对应的进程重新进行更新。

 ![image-20250830171723235](https://i-blog.csdnimg.cn/img_convert/a8c03b2eb0acd12ab6c66fbd89734c5b.png)

```
#更新软件源列表
sudo apt-get update
#更新软件版本
sudo apt-get upgrade
sudo apt-get dist-upgrade
```

更新的过程会比较慢，耐心等待即可。

 ![image-20250818011243614](https://i-blog.csdnimg.cn/img_convert/b162dc09ce306453edd74250e7c3866f.png)

下面是安装完毕之后的截图。

然后pip设置一下清华的镜像，方便后面安装python的库使用。

```bash
 pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

##### 检查摄像头和python环境

首先是摄像头的检查，由于我们安装的版本比较新，在新版本中已经丢弃了原先的raspistill命令，如果您看了比较老的教程，就可以直接放弃这个命令了。

 ![img](https://i-blog.csdnimg.cn/img_convert/c94cea8ce0a00ef7e9a76501adf8619a.png)

在我们的这个版本中，我们需要通过libcam-hello来对我们的摄像头进行测试，首先安装对应的软件，请执行下面的指令。

```bash
sudo apt install -y libcamera-dev libepoxy-dev libjpeg-dev libtiff5-dev libpng-dev libopencv-dev
sudo apt install libcamera-apps
sudo apt install libcamera-tools
```

执行完成之后，可以对摄像头进行一个简单的测试。( **注意：由于需要显示图像，这里的测试请通过vnc连接树莓派的方式来进行完成** )

rpicam-jpeg是一个简单的静态图像捕捉应用程序。要捕获全分辨率 JPEG 图像，请使用以下指令，这将显示大约五秒钟的预览，然后将全分辨率 JPEG 图像捕获到文件中test.jpg。

```
rpicam-jpeg -o test.jpg # 如果你是可视化屏幕则执行这个
```

 ![image-20250830190336090](https://i-blog.csdnimg.cn/img_convert/953e2666200c5029e9ad09aa0e699bda.png)

如果您是命令行执行的测试，那需要关闭预览窗口，因为无法通过mobaxterm进行转化，关闭窗口直接拍摄需要使用命令 `-n` ，如下所示。

```
rpicam-jpeg -o test.jpg -n
```

 ![image-20250828163627241](https://i-blog.csdnimg.cn/img_convert/015ab1cde45d9397aa12557cdc574bcc.png)

下面是拍摄之后的结果。

 ![image-20250828163702649](https://i-blog.csdnimg.cn/img_convert/2401becde8701f47649b20d18db4d79e.png)

> 其余参考资料：
> 
> 如果想要了解更多的libcamera的用法，请参考： [基于Raspberry 的 libcamera 使用 - 小淼博客 - 博客园](https://www.cnblogs.com/uestc-mm/p/17442508.html) 
> 
> 如果您不是树莓派的官方摄像头，还需要对一些参数进行设置，参数的设置参考这里： [树莓派5使用CSI摄像头 | 树莓派5接上CSI摄像头后命令行操作显示检测无设备_树莓派5 摄像头-CSDN博客](https://blog.csdn.net/beyourself_he/article/details/140896207) 
> 
> 或者如果您想要了解到关于树莓派摄像头更多的信息请看这里： [摄像头软件 | Raspberry Pi 树莓派 (官网25年6月更新)](https://pidoc.cn/docs/computers/camera-software/) 

在命令行中输入python确定python版本为3.11

 ![image-20250818112738038](https://i-blog.csdnimg.cn/img_convert/0fc430932e25404e0dd48bba0937014e.png)

#### 在树莓派上使用YOLO

首先需要将我们需要使用到的源码和轮子文件上传到树莓派上，我们这里使用moba来进行上传，首先我们需要先建立一个文件夹。

 ![image-20250818113352236](https://i-blog.csdnimg.cn/img_convert/e0064cdb406096ae893a1586177e3280.png)

安装python的库之前需要先解决一个bug，“externally-managed-environment”错误背后的原因：Manjaro、Ubuntu、Fedora 以及其他的最新发行版中，正在使用 Python 包来实现此增强功能。
这个更新是为了避免「操作系统包管理器 (如pacman、yum、apt) 和 pip 等特定于 Python 的包管理工具之间的冲突」。这些冲突包括 Python 级 API 不兼容和文件所有权冲突。

 ![image-20250830174855910](https://i-blog.csdnimg.cn/img_convert/8db31e479ae74408717f5be40999b28a.png)

执行下列的指令即可, 如果你不喜欢这个方式，这里还有另外的可以参考的方式： [深海游弋的鱼 – 默默的点滴](https://www.mobibrw.com/2024/41745) 

```
sudo mv /usr/lib/python3.11/EXTERNALLY-MANAGED /usr/lib/python3.11/EXTERNALLY-MANAGED.bak
```

然后将文件拖到这个文件夹即可，然后我们将已经编辑好的轮子文件进行安装即可，注意安装的时候你需要进入到对应的目录安装，安装执行下列指令，表示这个目录下的所有whl文件都安装，注意安装的时候需要按照主播这里提供的顺序安装，如果您不是按照顺序来安装的，会导致出现numpy的错误。

```bash
pip install cycler-0.12.1-py3-none-any.whl
pip install mpmath-1.3.0-py3-none-any.whl
pip install contourpy-1.3.3-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl
pip install fonttools-4.59.2-cp311-cp311-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl
pip install numpy-2.2.5-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install matplotlib-3.10.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install networkx-3.5-py3-none-any.whl
pip install PyYAML-6.0.2rc1-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install sympy-1.14.0-py3-none-any.whl
pip install scipy-1.9.3-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install opencv_python-4.12.0.88-cp37-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
pip install torch-2.0.1-cp311-cp311-manylinux2014_aarch64.whl
pip install torchvision-0.15.2-cp311-cp311-manylinux2014_aarch64.whl
pip install torch-2.0.1-cp311-cp311-manylinux2014_aarch64.whl
pip install tqdm-4.59.0-py2.py3-none-any.whl
pip install shiboken6-6.6.3.1-cp38-abi3-manylinux_2_31_aarch64.whl
pip install PySide6_Addons-6.6.3.1-cp38-abi3-manylinux_2_31_aarch64.whl
pip install PySide6-6.6.3.1-cp38-abi3-manylinux_2_31_aarch64.whl
pip install numpy-1.23.5-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

 ![image-20250818150912296](https://i-blog.csdnimg.cn/img_convert/fd2edb95b0ce4d223f0156093463904a.png)

安装完成之后测试一下torch是否已经安装成功，如果正确输出版本信息，说明torch的安装没有问题。

 ![image-20250818151117794](https://i-blog.csdnimg.cn/img_convert/181b8a803f9ebb1f188a5f8e3b041fcb.png)

如果您需要下载一些额外的库，可以去这个网站上寻找：https://pypi.tuna.tsinghua.edu.cn/simple/

torch安装完成之后，我们还需要安装一个摄像头拍摄的库，这个库为picamera2，注意这个库不能通过pip的方式来进行安装，需要通过apt install的方式来进行安装，如下所示，执行下列的命令，分别包含了qt的窗口界面以及picamera2这个库。

```
sudo apt-get update
sudo apt-get install libxcb-cursor0
sudo apt install -y python3-picamera2
sudo apt install -y python3-pyqt5 python3-opengl
```

首先我们有一个测试脚本 `test.py` ，这个测试脚本可以测试是否通过python命令的形式来调用摄像头，脚本的内容如下。

```python
import time
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)
encoder = H264Encoder(10000000)
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)
picam2.start_preview(Preview.QTGL) 
picam2.start_recording(encoder, 'test.h264') 
time.sleep(10)
picam2.stop_recording() 

```

执行下面的代码，如果摄像头可以正常地调用，说明你的摄像头是没有问题的，可以继续执行下面的步骤。

 ![image-20250830144404191](https://i-blog.csdnimg.cn/img_convert/af33649bfa36c8508e90fc8fa713c96d.png)

安装完成之后，将代码上传到我们的系统中，其中 `main_window_pi.py` 是我们的主界面文件，执行执行下列指令启动我们的主界面文件即可。

```bash
python main_window_pi.py
```

界面启动之后你可以选择上传图片进行测试，或者是通过直接打开摄像头的方式进行测试，如下所示，是通过摄像头的方式进行测试，测试结果如下。

 ![image-20250830144033643](https://i-blog.csdnimg.cn/img_convert/d75cfdbe6b5a0d6625b4ff37803bdd6e.png)

查看邮件，邮箱中收到了对应的火灾的邮件信息。

 ![image-20250830151357267](https://i-blog.csdnimg.cn/img_convert/dbf35cc0cd85f28ae2a2ce97dfeb4d7a.png)

如果你要关闭树莓派，请执行下列指令

```
sudo shutdown -h now
```

### 番外 使用网线进行树莓派的连接

有的时候可能您没有对应的显示器，或者是您的wifi的环境发生了变化，这个时候不知道树莓派的ip地址，可以考虑使用网线进行树莓派的连接。

首先您需要通过网线将树莓派和你的主机或者笔记本连接起来之后开机，如下图所示。

 ![image-20250818104706123](https://i-blog.csdnimg.cn/img_convert/7ca53246d2c65be3939d2d4dcaa25aa0.png)

连接之后，首先需要你对你电脑的wifi进行设置。

点击这里的编辑。

 ![image-20250818104926812](https://i-blog.csdnimg.cn/img_convert/3f0ba63ec10fae58194e48c83146473c.png)

在共享中点击允许，这个是为了让我们的电脑可以将网络共享给树莓派。

 ![image-20250818105039621](https://i-blog.csdnimg.cn/img_convert/f52a0a84109edb9ee3f3cefa66a728f2.png)

保存之后找到以太网。

 ![image-20250818105132428](https://i-blog.csdnimg.cn/img_convert/0c9ad57e0622e2a8c42eca4cd6de2136.png)

对ip地址进行编辑

 ![image-20250818145040889](https://i-blog.csdnimg.cn/img_convert/983ae00206e2d7197308590f12ad30a9.png)

 ![image-20250818145154341](https://i-blog.csdnimg.cn/img_convert/a871aadbda04278d1c8d3c08c154d980.png)

或者如果你是其他的windows系统，你可以使用这种方式来进行修改。

修改确定之后在cmd中输入下列指令

 ![image-20250818145227781](https://i-blog.csdnimg.cn/img_convert/daca22b39164d240aa8a1a173d3cd159.png)

然后找到我们网口的ip地址

 ![image-20250818145309537](https://i-blog.csdnimg.cn/img_convert/13c0df70a4138b7f6acf646e3136c890.png)

详细的解释可以参考：https://blog.csdn.net/Mr_zhubao/article/details/135876595