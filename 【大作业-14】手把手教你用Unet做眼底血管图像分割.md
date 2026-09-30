# 【大作业-14】手把手教你用Unet做眼底血管图像分割

## 手把手教你用Unet做眼底血管图像分割

> 配套教学视频地址： [手把手教你用Unet做眼底血管图像分割_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1UC411s7im/) 

> 配套代码和数据下载地址： [Unet眼底血管图像分割数据集+代码+模型+系统界面+教学视频.zip资源-CSDN文库](https://download.csdn.net/download/ECHOSON/88840844) 

Hi，这里是肆十二，今天我们来继续医学方向的大作业更新，今天选用的题材是基于Unet的眼底血管图像分割，废话不多说，先看实验结果。

 ![image-20240217143126005](https://i-blog.csdnimg.cn/blog_migrate/1bac18a28813e7a22ca401fab5bb5fdb.png)

 ![](https://i-blog.csdnimg.cn/blog_migrate/ec118e768a59cdd84d1a3819ddde65ba.png)

 ![mIoU](https://i-blog.csdnimg.cn/blog_migrate/b048de82c28c4e48c018b2efbed4fb4c.png)

之后我将会从原理、数据、训练、测试和界面封装四个章节对整体内容进行介绍。

**背景和意义** ：随着生活水平的提高，眼科疾病以及心脑血管疾病的发病率呈现逐年增长的趋势。视网膜血管是这类疾病诊断和监测的重要信息来源，其形态和状况的变化可以反映出许多疾病的早期病理变化。然而，由于受眼底图像采集技术的限制以及视网膜血管自身结构的复杂性和多变性，使得视网膜血管的分割变得非常困难。传统方法依靠人工手动分割视网膜血管，不仅工作量巨大，极为耗时，而且受主观因素影响严重。通过眼底血管图像分割可以提高诊断准确性、效率以及推动科学研究和改进治疗方法等方面。

### Unet网路结构介绍

细节方面还是比较推荐大家看原始论文，我在压缩包中也放置了原始论文和原始论文的翻译，在压缩包的这个位置，原汁原味得到的东西才是最真实的。

如下图所示，是Unet的网络结构图。

 ![image-20240216204709786](https://i-blog.csdnimg.cn/blog_migrate/e9dff12e21b0bc06c791594166ebdeb2.png)

U-Net是一种全卷积神经网络（Fully Convolutional Network，FCN），最初于2015年提出，主要应用于医学图像分割领域。U-Net的网络结构是对称的，形状类似于英文字母“U”，因此得名。

U-Net主要由两部分组成：左侧的特征提取部分（编码器，Encoder）和右侧的特征融合部分（解码器，Decoder）。左侧的特征提取部分通过一系列的卷积和下采样操作来提取输入图像的抽象特征。这些操作可以有效地降低图像的维度，同时保留重要的空间信息。右侧的特征融合部分则通过上采样和特征拼接操作来逐渐恢复图像的细节信息，并实现对像素级别的分类。

在U-Net中，左侧的特征提取部分和右侧的特征融合部分是通过跳跃连接（Skip Connection）进行连接的。具体来说，左侧每个下采样模块的输出都会与右侧对应上采样模块的输出进行拼接，这样可以将浅层的细节信息和深层的抽象信息有效地结合起来，提高分割的精度。

与传统的FCN相比，U-Net使用了特征拼接而非简单的特征相加来实现特征的融合。这种特征拼接方式可以形成更厚的特征图，从而更充分地利用图像的上下文信息。同时，U-Net的对称结构也使得特征的融合更加彻底。

### 数据准备

模型训练开始之前，我们需要准备好训练和测试使用的原始数据和标签。

注：此处的教程使用标签为黑白的2分类模型的训练，多分类或者标签不是黑白的无法进行训练。

在数据集目录下新建下面的四个文件夹，分别用于放置训练和测试使用的原始图像和标签图像。

 ![image-20240217003228369](https://i-blog.csdnimg.cn/blog_migrate/86f63e3beea3a482a365edaba93be93d.png)

以训练集中的一张图像为例，需要保证原始图像和标签图像的名称一致，并且标签图像为黑白，白色表示需要进行分割的目标区域，黑色则表示背景区域。测试集同理。

 ![image-20240217003422421](https://i-blog.csdnimg.cn/blog_migrate/96f467256330bcf6f5b33a8afc7959ef.png)

### 环境配置

老规矩，首先还是需要我们下载本教程使用的代码和数据集，压缩包下载之后，请将其解压在一个英文的路径下面，因为中文路径可能会导致图片读取错误。

压缩包下载地址为： [Unet眼底血管图像分割数据集+代码+模型+系统界面+教学视频.zip资源-CSDN文库](https://download.csdn.net/download/ECHOSON/88840844) 

 ![image-20240217152505954](https://i-blog.csdnimg.cn/blog_migrate/b8d624c85db50f53bf9f8b4715086556.png)

解压之后的文件夹如下所示：

 ![image-20240217152609493](https://i-blog.csdnimg.cn/blog_migrate/9ec3d2e17940bd5c6b82ffebe6d63a64.png)

安装好Anaconda和Pycharm之后，如果没有安装的小伙伴请看这期教程： [【2024年毕设系列】如何使用Anaconda和Pycharm-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/136097910?spm=1001.2014.3001.5502) 

在项目目录下执行下列一系列指令即可。

 ![image-20240217152810180](https://i-blog.csdnimg.cn/blog_migrate/6f0a90d8a7d9287b757edfdc100e3d35.png)

- 建立虚拟环境

  ```bash
  conda create -n drive python==3.8.5
  ```

- 激活虚拟环境

  ```bash
  conda activate drive
  ```

- 安装Pyotorch

  首先点击设备管理器查看你本地的电脑是否具有Nvidia显卡

   ![image-20240217153141046](https://i-blog.csdnimg.cn/blog_migrate/d203ef57628d1384c22b930af06a2f35.png)

  有显卡的小伙伴执行下列指令

  ```bash
  conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 
  ```

  没有显卡的小伙伴执行下列指令

  ```bash
  conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly
  ```

  我这边有显卡，所以我执行这个指令

   ![image-20240217153307504](https://i-blog.csdnimg.cn/blog_migrate/da47e2abd951118c60224ee543c537c6.png)

- 安装其他依赖

  安装Pytorch需要花费比较多的时间，安装过程中如果比较慢的话大家也执行通过手机热点的形式来进行安装。

  之后执行下列指令安装本项目所需要的其他依赖。

  ```bash
  pip install -r requirements.txt
  ```

- 使用Pycharm打开项目并激活你建立的虚拟环境

  使用Pycharm打开你的项目之后再pycharm的右下角选择您上面建立的虚拟环境，配置就完成了。

   ![image-20240217154317605](https://i-blog.csdnimg.cn/blog_migrate/3c0b637479fcd7a172a8e42ab72347bc.png)

下面我们对每个环节进行详细讲解。

### 模型训练

模型训练部分，其中图像的输入为大小为512的灰度图像。

**模型的损失函数部分使用的是BCELoss** 。PyTorch中的BCELoss指的是Binary Cross Entropy Loss，即二值交叉熵损失，它适用于0/1二分类问题。这个损失函数的主要作用是在训练神经网络时，衡量模型预测的输出与真实的标签之间的差异。计算公式是“-ylog(y_hat) - (1-y)log(1-y_hat)”，其中y是实际的标签（ground truth），y_hat是模型的预测值。在这个公式中，当y为0时，公式的前半部分为0，此时y_hat需要尽可能接近0才能使后半部分的数值更小；当y为1时，公式的后半部分为0，此时y_hat需要尽可能接近1才能使前半部分的数值更小。这样，通过优化BCELoss，我们可以使模型的预测值y_hat尽可能地接近真实的标签y。此外，使用BCELoss需要注意，网络的输出需要在0-1之间。为了实现这一点，通常会在网络的输出层添加一个Sigmoid函数，将输出值映射到0-1的范围内。

**模型的优化算法部分使用的是RMSprop** 。RMSprop算法是一种自适应学习率的优化算法，由Geoffrey Hinton提出，主要用于解决梯度下降中的学习率调整问题。在梯度下降中，每个参数的学习率是固定的，但实际应用中，每个参数的最优学习率可能是不同的。如果学习率过大，则模型可能会跳出最优值；如果学习率过小，则模型的收敛速度可能会变慢。RMSprop算法通过自动调整每个参数的学习率来解决这个问题。具体来说，RMSprop算法在每次迭代中维护一个指数加权平均值，用于调整每个参数的学习率。如果某个参数的梯度较大，则RMSprop算法会自动减小它的学习率；如果梯度较小，则会增加学习率。这样可以使得模型的收敛速度更快。（这里大家可以尝试修改为Adam算法，收敛更快）。

大家按照下图设置好数据集的路径、基础的学习率以及训练时候的批次大小，就能展开训练了。

 ![image-20240216234745951](https://i-blog.csdnimg.cn/blog_migrate/b88b5c70639a0eff326e72eb8b32f762.png)

训练之后模型保存在项目的当前目录下，名称为 `best_model.pth` ，并且会生成对应的训练过程中的损失变化折线图，你可以通过折线图的趋势来观察是否模型是否收敛。

 ![image-20240217000954803](https://i-blog.csdnimg.cn/blog_migrate/7ecc260c3602b75d0d3301cdbf70a96f.png)

这里需要记住模型的路径以方便我们后期测试和图形化界面程序使用。

### 模型测试

模型测试部分我们使用语义分割常用的四个指标，分别是precision、recall、mPA和mIoU

- precision

  在语义分割中，Precision（精确度，也称为查准率）是一个重要的评价指标。它衡量的是模型预测为正例的样本中，真正为正例的样本所占的比例。换句话说，Precision表示的是“模型预测为正例的样本中有多少是真正的正例”。

  Precision的计算公式为：Precision = TP / (TP + FP)，其中TP表示真正例（True Positive），即模型预测为正例且实际也为正例的样本数；FP表示假正例（False Positive），即模型预测为正例但实际为负例的样本数。

  在语义分割任务中，通常将像素点作为样本进行处理，因此Precision指标可以用来衡量模型对于正例像素点的预测准确性。需要注意的是，在实际应用中，可能会根据具体任务和数据集的特点对Precision指标进行一定的调整或变种。

- recall

  在语义分割中，Recall（召回率，也称为查全率）是一个重要的评价指标，用于衡量模型找出真正正例样本的能力。具体来说，Recall计算的是所有真实标签为正例的样本中，被模型正确预测为正例的样本所占的比例。

  Recall的计算公式为：Recall = TP / (TP + FN)，其中TP表示真正例（True Positive），即模型预测为正例且实际也为正例的样本数；FN表示假反例（False Negative），即模型预测为负例但实际为正例的样本数。

  在语义分割任务中，通常将图像的像素作为样本，因此Recall指标可以用来衡量模型对于正例像素点的检测能力。如果模型的Recall值较高，说明模型能够较好地找出图像中的正例像素点，即分割结果更加完整。

  需要注意的是，Recall和Precision是两个相互制约的指标，通常情况下提高Recall会导致Precision的下降，反之亦然。因此，在评估语义分割模型性能时，需要综合考虑这两个指标，并找到它们的平衡点。此外，还可以使用F1-score等指标来综合考虑Recall和Precision的表现。

- mIoU

  语义分割中的mIoU（Mean Intersection over Union）是一种常用的评价指标，用于衡量分割结果的准确性。mIoU计算的是真实值和预测值两个集合的交集和并集之比，这个比例可以变形为TP（交集）比上TP、FP、FN之和（并集）。在每个类别上计算IoU，然后取平均，即可得到mIoU。

  具体来说，对于每个类别，我们都可以将预测结果和真实结果看作是两个集合，然后计算这两个集合的交集和并集。交集部分表示预测正确的像素点数量，并集部分表示真实值或预测值中存在的像素点数量。IoU就是交集和并集的比值，用来衡量预测结果和真实结果的相似程度。

  在语义分割任务中，mIoU越高表示预测结果越准确，因为这意味着预测值和真实值的交集部分越大，同时并集部分越小，即预测错误的像素点越少。

  需要注意的是，mIoU是一种基于像素级别的评价指标，它只关注像素点的分类结果是否正确，而不考虑像素点之间的空间关系。因此，在某些情况下，mIoU可能无法完全反映分割结果的优劣。为了更全面地评估分割结果，可能需要结合其他指标进行评价。

- mPA

  在语义分割中，mPA（Mean Pixel Accuracy，均像素精度）通常不是一个标准的评价指标。更常见的是MPA（Mean Per-class Accuracy，平均类别像素准确率），它衡量的是每个类别内被正确分类的像素比例的平均值，以及MIoU（Mean Intersection over Union，平均交并比），这是语义分割中最常用的指标之一。

  然而，如果你提到的mPA是指的均像素精度，并且想要了解类似的概念，那可能是对所有像素的准确率的某种平均，但在实践中，这并不是一个常用的度量标准，因为它没有考虑到类别的不平衡问题。

  通常所说的MPA（平均类别像素准确率）是按类别计算的像素准确率的平均值，它的计算方法如下：

  1. 对于每个类别，计算该类别的像素准确率，即正确分类的像素数除以该类别的总像素数。

  2. 将所有类别的像素准确率相加。

  3. 将总和除以类别的数量，得到平均类别像素准确率（MPA）。

  这个指标考虑了每个类别的表现，但仍然可能受到类别不平衡的影响。

OK，理论的部分解释完毕，下面进入到实操部分，测试部分我们通过step2_test.py来进行完成

 ![image-20240217001315337](https://i-blog.csdnimg.cn/blog_migrate/f4d2995a2033979bafecba654dc105f4.png)

来看一下我们眼底血管图像执行的结果吧。

 ![image-20240217001423662](https://i-blog.csdnimg.cn/blog_migrate/07d364104f5244e583ef11808308e0e8.png)

另外，我在这里专门写了一个用于批量预测的脚本 `step3_predict.py` ，大家也可以根据批量预测的脚本完成一些自己的定制化开发需求。

 ![image-20240217003015303](https://i-blog.csdnimg.cn/blog_migrate/7669a379ede53b36d14906252992d43f.png)

### 图形化界面构建

图形化界面部分我们使用的Pyqt5，同时支持视频和图像的检测，视频由于没有临床的数据，我这边用图片合成了一个视频放在我们的项目目录下，其中源码如下。

```python
# -*-coding:utf-8 -*-

"""
#-------------------------------
# @Author : 肆十二
# @QQ : 3045834499 可定制毕设
#-------------------------------
# @File : step4_window.py
# @Description: 图形化界面，支持图片检测和视频检测，并输出对应的占比
# @Software : PyCharm
# @Time : 2024/2/14 10:48
#-------------------------------
"""

import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from model.unet_model import UNet
import numpy as np
import time
import threading

# 窗口主类
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 需要添加视频预测的部分并且临时关闭
class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('基于Unet的眼底血管分割 99远程调试+Q：3045834499')
        # self.setStatusTip("远程调试请联系qq：304834499")
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        # # 初始化视频读取线程
        self.origin_shape = ()
        # 加载网络，图片单通道，分类为1。
        #  初始化视频检测相关的内容
        self.vid_source = 'demo.mp4' # 要进行视频检测的名称
        self.video_capture = cv2.VideoCapture(self.vid_source)
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()

        net = UNet(n_channels=1, n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('best_model.pth', map_location=device))  # todo 模型位置
        # 测试模式
        net.eval()
        self.model = net
        self.initUI()

    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        # img_detection_title = QLabel("图片识别功能" + "\n99远程调试+Q：3045834499")
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setAlignment(Qt.AlignCenter)
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # 添加视频检测的页面
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
        # vid_title = QLabel("视频检测功能" + "\n99远程调试+Q：3045834499")
        vid_title = QLabel("视频检测功能")
        vid_title.setFont(font_title)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        vid_title.setAlignment(Qt.AlignCenter)
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        self.webcam_detection_btn.setStyleSheet("QPushButton{color:white}"
                                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                                "QPushButton{background-color:rgb(48,124,208)}"
                                                "QPushButton{border:2px}"
                                                "QPushButton{border-radius:5px}"
                                                "QPushButton{padding:5px 5px}"
                                                "QPushButton{margin:5px 5px}")
        self.mp4_detection_btn.setStyleSheet("QPushButton{color:white}"
                                             "QPushButton:hover{background-color: rgb(2,110,180);}"
                                             "QPushButton{background-color:rgb(48,124,208)}"
                                             "QPushButton{border:2px}"
                                             "QPushButton{border-radius:5px}"
                                             "QPushButton{padding:5px 5px}"
                                             "QPushButton{margin:5px 5px}")
        self.vid_stop_btn.setStyleSheet("QPushButton{color:white}"
                                        "QPushButton:hover{background-color: rgb(2,110,180);}"
                                        "QPushButton{background-color:rgb(48,124,208)}"
                                        "QPushButton{border:2px}"
                                        "QPushButton{border-radius:5px}"
                                        "QPushButton{padding:5px 5px}"
                                        "QPushButton{margin:5px 5px}")
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        vid_detection_layout.addWidget(vid_title)
        vid_detection_layout.addWidget(self.vid_img)
        # todo 添加摄像头检测标签逻辑
        # self.vid_num_label = QLabel("当前检测结果：{}".format("等待检测"))
        # self.vid_num_label.setFont(font_main)
        # vid_detection_layout.addWidget(self.vid_num_label)
        # 直接展示的时候分成左边和右边进行展示比较方便一些
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel(
            '欢迎使用医学影像语义分割系统\n\n 提供付费指导：有需要的好兄弟加下面的QQ即可')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/qq.png'))
        about_img.setAlignment(Qt.AlignCenter)

        # label4.setText("<a href='https://oi.wiki/wiki/学习率的调整'>如何调整学习率</a>")
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>或者你可以在这里找到我-->肆十二</a>")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')
        self.addTab(about_widget, '联系我')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))
        self.setTabIcon(2, QIcon('images/UI/lufei.png'))

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))

    '''
    ***检测图片***
    '''

    def detect_img(self):
        # 视频在这个基础上加入for循环进来
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        img = cv2.imread(source)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (512, 512))
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = self.model(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        im0 = cv2.resize(pred, self.origin_shape)
        cv2.imwrite("images/tmp/single_result.jpg", im0)
        # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

        # 界面关闭

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    # 添加摄像头实时检测的功能，界面和一个可以使用的for循环界面
    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = '0'
        self.video_capture = cv2.VideoCapture(self.vid_source)
        self.webcam = True
        th = threading.Thread(target=self.detect_vid)
        th.start()

    # 视频文件检测
    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName # 这个里面给定的需要进行检测的视频源
            self.video_capture = cv2.VideoCapture(self.vid_source)
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    # 视频检测主函数
    def detect_vid(self):
        # model = self.model
        # 加载模型 不断从源头读取数据
        while True:
            ret, frame = self.video_capture.read()  # 读取摄像头
            if not ret:
                self.stopEvent.set()
                # break  # 如果读取失败（例如，已经到达视频的结尾），则退出循环
            else:
                # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
                # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像转化为rgb颜色通道
                ############### todo 加载送入模型进行检测的逻辑， 以frame变量的形式给出
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (512, 512))
                # 转为batch为1，通道为1，大小为512*512的数组
                img = img.reshape(1, 1, img.shape[0], img.shape[1])
                # 转为tensor
                img_tensor = torch.from_numpy(img)
                # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
                img_tensor = img_tensor.to(device=device, dtype=torch.float32)
                # 预测
                pred = self.model(img_tensor)
                # 提取结果
                pred = np.array(pred.data.cpu()[0])[0]
                # 处理结果
                pred[pred >= 0.5] = 255
                pred[pred < 0.5] = 0
                # 保存图片
                # im0 = cv2.resize(pred, self.origin_shape)

                # frame = frame
                frame_height = frame.shape[0]
                frame_width = frame.shape[1]
                frame_scale = self.output_size / frame_height
                frame_resize = cv2.resize(pred, (int(frame_width * frame_scale), int(frame_height * frame_scale)))
                src_frame = cv2.resize(frame, (int(frame_width * frame_scale), int(frame_height * frame_scale)))
                # src_frame = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
                # 合成完毕之后，在颜色通道上进行转化
                frame_resize_RGB = cv2.cvtColor(frame_resize, cv2.COLOR_GRAY2RGB)
                hstack_result = np.hstack((src_frame, frame_resize_RGB))
                cv2.imwrite("images/tmp/tmp.jpg", hstack_result)
                # 展示图片的时候，应该将frame的图片和原始图片进行合并，合并只是
                self.vid_img.setPixmap(QPixmap("images/tmp/tmp.jpg"))
            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.vid_img.clear()
                self.vid_stop_btn.setEnabled(False)
                self.webcam_detection_btn.setEnabled(True)
                self.mp4_detection_btn.setEnabled(True)
                self.reset_vid()
                break

    # 摄像头重置
    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        # self.vid_source = self.init_vid_id
        self.webcam = True
        self.video_capture.release()
        cv2.destroyAllWindows()
        # self.vid_num_label.setText("当前检测结果：{}".format("等待检测"))

    # 视频线程关闭
    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

```

图像和视频的测试结果如下。

 ![image-20240217142956972](https://i-blog.csdnimg.cn/blog_migrate/57db9a90fc49b88b6269177c17f4932a.png)

 ![image-20240217143027426](https://i-blog.csdnimg.cn/blog_migrate/4ba228bc919cd0e26749b03a22af1b82.png)

有问题以及需要远程调试可以通过CSDN私信联系我！