# 【大作业-15】手把手教你用Unet++做舌象分割

## 【2024年毕设系列】手把手教你用Unet++做舌象分割

> 视频教程地址： [教你用Unet++做舌象分割_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1wJ4m1a7cJ/?vd_source=2f9a4e63109c3db3be5e8078e5111776) 
> 
> 资源地址： [Unet++舌象图像分割数据集+代码+模型+系统界面+教学视频.zip资源-CSDN文库](https://download.csdn.net/download/ECHOSON/88938463) 

Hi，各位好久不见，这里是肆十二，转眼现在已经到了3月份的中旬。

今天我们讨论的主题依然是医学图像分割，今天选用的网络结构是unet的改进版本unet++。

UNet++，它是一种深度监督的编码器-解码器网络，通过一系列嵌套的密集跳跃连接将编码器和解码器子网连接起来。UNet++的设计目标是减少编码器和解码器子网特征图之间的语义差距，使得优化器在面对语义相似的特征图时，学习任务变得更加简单。

废话不多说，先看实现效果，我们将会对模型的训练测试以及推理进行原理介绍和代码实现，最终你将会利用unet++的这个网络结构训练一个舌象分割的模型，上传图片或者视频可以利用训练好的模型对舌象进行预测并给出分割结果。

 ![image-20240310122222021](https://i-blog.csdnimg.cn/blog_migrate/fb8beb0909e138b07b75449b31c81789.png)

 ![](https://i-blog.csdnimg.cn/blog_migrate/437963e9055e1d910e4970694694503a.png)

### 1. Unet++网络结构介绍

#### 1.1 论文解读

原始的论文放置在DOCS目录下，其中 `unet++.pdf` 的文件是该网络结构的原始论文。

本文介绍了一种新的医学图像分割架构——UNet++，它是一种深度监督的编码器-解码器网络，通过一系列嵌套的密集跳跃连接将编码器和解码器子网连接起来。UNet++的设计目标是减少编码器和解码器子网特征图之间的语义差距，使得优化器在面对语义相似的特征图时，学习任务变得更加简单。

与现有的U-Net和宽U-Net架构相比，UNet++在多个医学图像分割任务中表现出色，包括低剂量CT扫描中的结节分割、显微镜图像中的细胞核分割、腹部CT扫描中的肝脏分割，以及结肠镜检查视频中的息肉分割。实验结果显示，UNet++通过深度监督实现了比U-Net和宽U-Net更高的平均IoU增益，分别提高了3.9和3.4个百分点。
总的来说，UNet++通过改进跳跃连接的设计，提高了医学图像分割的性能。这一创新在编码器-解码器架构中具有重要的应用前景，有助于解决医学图像分割中的挑战，如目标对象的精细细节恢复和复杂背景下的准确分割。此外，UNet++的成功也验证了深度监督在改善网络性能方面的有效性，为未来的图像分割研究提供了新的思路和方法。

#### 1.2 网络结构概述

Unet++是一种基于深度监督的编码器-解码器网络结构，该结构在原始的U-Net基础上进行了改进，通过引入密集的跳跃连接和重新设计的跳跃路径，以及深度监督机制，来提高图像分割的精度。

Unet++的网络结构主要由编码器、解码器和跳跃连接组成。编码器负责将输入图像转化为特征，解码器则负责将这些特征还原为输出结果。在编码器和解码器之间，通过跳跃连接将深层的特征信息与浅层的特征信息进行融合，从而结合不同尺度的特征，提高分割精度。

与原始的U-Net相比，Unet++在跳跃连接上进行了改进，采用了密集的跳跃连接，使得每个解码器层都能接收到来自编码器层的特征信息。此外，Unet++还重新设计了跳跃路径，通过添加额外的卷积层来减少编码器和解码器之间的语义差距，进一步优化了网络结构。

在深度监督方面，Unet++采用了多个分支进行输出，并根据不同的模式（如精确模式和快速模式）选择不同的分支进行训练。这种机制可以使网络在训练过程中更加稳定，并提高分割精度。

总之，Unet++的网络结构通过引入密集的跳跃连接、重新设计的跳跃路径和深度监督机制等改进，使得网络在图像分割任务中具有更高的精度和稳定性。

 ![image-20240310111959919](https://i-blog.csdnimg.cn/blog_migrate/68790d69a632f7b932a88b0269c08ee0.png)

该图是论文中给出的Unet++的网络结构图，其中图a表示的是Unet++的网络结构，从网络结构里面可以看出采用了密集连接的方式，并且图b给出了该跳跃链接的具体链接方式，中间的神经元接受来自多个方向的输出。训练的过程中采用图a的完全体的结构，通过这种完全体的结构可以输出四个尺度预测结果，在四个尺度上的预测结果进行分别loss的计算对网络参数进行优化，在实际进行预测和推理的时候，则可以对网络结构进行减枝，也就是图c中演示的，通过减少其他子结构的输出来进行推理，可以有效地降低网络在训练过程中的参数量。

这种链接方式的优势是：可以抓取不同层次的特征,将它们通过特征叠加的方式整合. 不同层次的特征，或者说不同大小的感受野，对于大小不一的目标对象的敏感度是不同的，比如，感受野大的特征，可以很容易的识别出大物体的，但是在实际分割中，大物体边缘信息和小物体本身是很容易被深层网络一次次的降采样和一次次升采样给弄丢的，这个时候就可能需要感受野小的特征来帮助.而UNet++就是拥有不同大小的感受野,所以效果好。

### 数据准备和环境配置

安装完成之后首先需要从CSDN上下载我这边为大家准备好的代码和数据：

> 资源下载地址为： [Unet++舌象图像分割数据集+代码+模型+系统界面+教学视频.zip资源-CSDN文库](https://download.csdn.net/download/ECHOSON/88938463) 

OK, 喝口水！环境配置启动！

 ![image-20240310123642262](https://i-blog.csdnimg.cn/blog_migrate/1ccfd3f33d2d318b07944d221b4566ba.png)

#### 数据集准备

数据集方面为了二分类的数据，其中原始图片为PNG格式的彩色图像，标签图像为PNG格式的黑白图像，如下图所示是训练集中原始图像和标签数据的展示。

 ![image-20240310121051350](https://i-blog.csdnimg.cn/blog_migrate/a0e6f653fed8361da6556f7888218a92.png)

#### 环境配置

环境配置之前依然需要各位提前学习并安装Anaconda和Pycharm，教程在这里： [【2024年毕设系列】如何使用Anaconda和Pycharm_blog.csdn.net/echoson/article/details/136097910-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/136097910) 

首先为了能够下载的比较快，先对国内镜像进行配置。

```bash
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
```

利用下面命令安装并激活虚拟环境

```bash
conda create -n unetpp python==3.8.5
conda activate unetpp
```

Pytorch需要使用conda命令单独安装，请根据自己的硬件状态进行选择。

```bash
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
```

进入到代码目录下，利用下面命令安装本代码所需要的其他依赖库

```bash
pip install -r requirements.txt
```

在Pycharm中加载配置好的虚拟环境

 ![image-20240310123744572](https://i-blog.csdnimg.cn/blog_migrate/e2a957ee7da03b603b677239bdffab26.png)

测试看看吧！执行 `unetpp_step4_window.py` !

如果正常输出界面表示你已经成功！

### 代码分析

#### 网络结构解析

网络结构部分我们使用Pyotorch完成，损失函数使用二进制交叉熵损失函数，网络结构的具体实现如下：

```python
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision

# unetpp原始论文：
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class UNetPP(nn.Module):
    def __init__(self, in_channel, out_channel, deepsupervision):
        """
                初始化UNetPP模型。
                :param in_channel: 输入图像的通道数。
                :param out_channel: 输出图像的通道数。
                :param deepsupervision: 是否使用深度监督。如果为True，则在不同层级上产生输出。
        """
        super().__init__()

        # self.args = args
        # 深度监督的标志
        self.deepsupervision = deepsupervision
        # 定义每个卷积层中的滤波器数量
        nb_filter = [32, 64, 128, 256, 512]
        # 定义最大池化层，用于下采样
        self.pool = nn.MaxPool2d(2, 2)
        # 定义上采样层，用于上采样
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 定义编码器部分的卷积层
        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])
        # 定义解码器部分的卷积层，同时融合编码器的特征图
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])
        # 定义更多的解码器卷积层，用于更深层次的特征融合
        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        # 定义Sigmoid激活函数，用于将输出映射到[0, 1]范围
        self.sigmoid = nn.Sigmoid()
        # 根据是否使用深度监督来定义输出层
        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        else:
            # 只在最后一个层级上产生输出
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    # 前向传播函数，根据上面初始化好的组件进行前向传播的操作
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output
```

#### 训练

模型训练部分的代码为 ` unetpp_step1_train.py` 文件。

我们需要从上级目录将舌象分割的数据集加载进来，之后初始化unetpp的网络结构，其中模型的输出通道为1，表示按照灰度图的形式进行加载，模型的输出通道为1，表示这是一个二分类的问题，另外，第三个参数为是否采用密集的跳跃链接，这里设定为True，表示采用密集的跳跃链接的结构。

主函数如下：

```python
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置输出的通道和输出的类别数目，这里的1表示执行的是二分类的任务
    net = UNetPP(1, 1, True).train()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "../TongeImageDataset"  # todo 或者使用相对路径也是可以的
    print("进度条出现卡着不动不是程序问题，是他正在计算，请耐心等待")
    time.sleep(1)
    train_net(net, device, data_path, epochs=40, batch_size=1)  # 开始训练，如果你GPU的显存小于4G，这里只能使用CPU来进行训练。

```

执行训练，将会在命令行中输出训练过程的损失变化，如果发现损失没有及时下降则需要调整学习率的大小。

 ![image-20240310115034757](https://i-blog.csdnimg.cn/blog_migrate/ce83bd8fcb9472b6b07c7e233fae6308.png)

下面所示是训练过程中loss的变化曲线图。

 ![training_loss](https://i-blog.csdnimg.cn/blog_migrate/1100161f5bdd63165df66b314e512fd0.png)

#### 测试

模型测试部分的代码为： ` unetpp_step2_test.py` 

```python
if __name__ == '__main__':
    cal_miou(test_dir="../TongeImageDataset/Test_Images",  # 测试集路径
             pred_dir="../TongeImageDataset/results",  # 测试集推理结果保存路径
             gt_dir="../TongeImageDataset/Test_Labels",  # 测试标签路径
             model_path='best_model_unetpp.pth')  # 训练好的模型路径
```

和训练不同的是，测试的代码输出测试的结果和指标，其中指标测试的结果将会保存在results目录下，如下图所示是本次测试输出的MIOU的指标结果图。

 ![image-20240310122406960](https://i-blog.csdnimg.cn/blog_migrate/437963e9055e1d910e4970694694503a.png)

#### 图形化界面

图形化界面采用Pyqt5开发，PYQT5是一个用于创建图形用户界面应用程序的Python绑定库，它基于Qt v5库。Qt本身是一个强大的C++库，广泛用于开发GUI应用程序以及跨平台软件和嵌入式应用。而PYQT5则为Python程序员提供了使用这个强大库的能力，使得Python开发者能够利用Qt的丰富功能和灵活性来构建复杂的图形用户界面。

我们使用PYQT5开发了两个子页面，这两个子页面可以完成对图像的检测和对视频的检测，视频检测部分的原理是分帧读取视频，将视频检测的问题转化为图像检测的问题，为了防止视频检测的时候程序闪退，这里使用多线程技术保证界面可以完美运行。入下图所示，是本文模型实际执行的图片检测效果图。

 ![image-20240310122222021](https://i-blog.csdnimg.cn/blog_migrate/fb8beb0909e138b07b75449b31c81789.png)

