# 【大作业收官-62】基于VGG的肺炎图像分类系统

## 【大作业收官-62】基于VGG的肺炎图像分类系统

[操作视频地址](https://www.bilibili.com/video/BV138Y9zMErG/?vd_source=2f9a4e63109c3db3be5e8078e5111776) 

### 前言

 ![image-20250817131745192](https://i-blog.csdnimg.cn/img_convert/9746ec07d5f3ef330b5696f32401bfec.png)

> 环境配置和软件安装： [Python项目配置前的准备工作-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/151867100?spm=1011.2415.3001.5331) 
> 
> 深度学习教程： [《动手学深度学习》 — 动手学深度学习 2.0.0 documentation](https://zh.d2l.ai/) 

我们在上一期的栏目中对卷积神经网络以及resnet进行了简单的介绍和使用，在resnet的实战过程中，resnet是通过imagenet这项比赛出名的，同样在这个比赛中还有一个谷歌提出的模型vgg也是非常经典，今天我们继续来看一下VGG这个网络结构。

### VGG网络介绍

虽然AlexNet证明深层神经网络卓有成效，但它没有提供一个通用的模板来指导后续的研究人员设计新的网络。 在下面的几个章节中，我们将介绍一些常用于设计深层神经网络的启发式概念。

与芯片设计中工程师从放置晶体管到逻辑元件再到逻辑块的过程类似，神经网络架构的设计也逐渐变得更加抽象。研究人员开始从单个神经元的角度思考问题，发展到整个层，现在又转向块，重复层的模式。

使用块的想法首先出现在牛津大学的 [视觉几何组（visual geometry group）](http://www.robots.ox.ac.uk/~vgg/) 的 *VGG网络* 中。通过使用循环和子程序，可以很容易地在任何现代深度学习框架的代码中实现这些重复的架构。

经典卷积神经网络的基本组成部分是下面的这个序列：

1. 带填充以保持分辨率的卷积层；

2. 非线性激活函数，如ReLU；

3. 汇聚层，如最大汇聚层。

而一个VGG块与之类似，由一系列卷积层组成，后面再加上用于空间下采样的最大汇聚层。在最初的VGG论文中 ( [Simonyan and Zisserman, 2014](https://zh.d2l.ai/chapter_references/zreferences.html#id153) )，作者使用了带有卷积核、填充为1（保持高度和宽度）的卷积层，和带有汇聚窗口、步幅为2（每个块后的分辨率减半）的最大汇聚层。在下面的代码中，我们定义了一个名为 `vgg_block` 的函数来实现一个VGG块。

```python
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

与AlexNet、LeNet一样，VGG网络可以分为两部分：第一部分主要由卷积层和汇聚层组成，第二部分由全连接层组成。

 ![image-20250817113035913](https://i-blog.csdnimg.cn/img_convert/c46d2ed923ccef322bcf89187f3c251b.png)

VGG神经网络连接的几个VGG块（在 `vgg_block` 函数中定义）。其中有超参数变量 `conv_arch` 。该变量指定了每个VGG块里卷积层个数和输出通道数。全连接模块则与AlexNet中的相同。

原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。 第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。

总的来说。

- VGG-11使用可复用的卷积块构造网络。不同的VGG模型可通过每个块中卷积层数量和输出通道数量的差异来定义。

- 块的使用导致网络定义的非常简洁。使用块可以有效地设计复杂的网络。

- 在VGG论文中，Simonyan和Ziserman尝试了各种架构。特别是他们发现深层且窄的卷积（即）比较浅层且宽的卷积更有效。

 ![image-20250817133909222](https://i-blog.csdnimg.cn/img_convert/b74d8ce785d8ad87322bf971efda0ca7.png)

### 项目实战

#### 数据的准备

数据集可以来源于你自己日常生活的拍照，或者是项目提供，或者是你直接从网络中进行下载获取，这里的数据最好进行清洗，保证你的数据集的名称中不包含中文以及你的数据集中的图片的通道数目为3，一方面是为了防止opecv在读取的过程中不会出现错误，另一方面是可以保证你的模型模型在训练的过程中不会因为通道数量的不匹配导致奇怪的中断。你所搜集的数据需要按照他们的类别放在不同的目录下面，比如玫瑰花的目录下面就需要全部是玫瑰花的图像，向日葵的目录下面则需要全部是向日葵的图像这样。

如果你的数据集中包含了训练集和测试集，则你不需要进行这一步，如果你的数据集中是一整个，则需要进行这一步。

在正式训练之前，你还需要准备好训练集、验证集和测试集，一般是需要将所有的数据按照6:2:2的比例进行划分。这里也为大家准备了 `data_split.py` 的脚本帮助大家做数据集的划分。划分之前，请先按照类别将对应的图片放在对应的子目录下，如下图所示：

 ![image-20250817114220352](https://i-blog.csdnimg.cn/img_convert/06d73eb74c800125d80ea729dc4ce42e.png)

然后使用 `data_split.py` 脚本即可，这里你只需要修改原始数据集的路径，直接运行就可以，之后将会产生一个后缀为split的目录，里面是按照6：2：2比例划分之后的数据集。

 ![image-20221129145655543](https://i-blog.csdnimg.cn/img_convert/29192a2e9e0498875e478e8ce6d03129.png)

 ![image-20221129145847033](https://i-blog.csdnimg.cn/img_convert/a7b91871fa88a8fa31f9f9301041b1f4.png)

如果你想要修改划分的比例，可以在这个位置进行修改，保证他们的和等于1即可。

 ![image-20221129145942981](https://i-blog.csdnimg.cn/img_convert/1afecaae4f656a1b97c63a794210b724.png)

记住这里的路径，我们将会在后续的训练和测试中经常用到。

 ![image-20221129150016346](https://i-blog.csdnimg.cn/img_convert/90f503ed080d48920013036a0d618bb6.png)

#### 模型的训练

为了保证大家在之后看视频的过程中也能够复现出来，我们提前对每个库的版本进行了固定，我们本次使用的pytorch的模型库为timm。 `timm` （PyTorch Image Models）是一个由 Ross Wightman 开发的高质量 PyTorch 图像模型库，广泛用于图像分类、特征提取和模型迁移学习任务。它包含了大量经典与最新的预训练模型，如 ResNet、EfficientNet、MobileNet、RegNet、Vision Transformer（ViT）、Swin Transformer、ConvNeXt、DeiT、CoAtNet 等，几乎涵盖了主流论文中的架构。 `timm` 提供统一且简洁的接口，使得模型加载和使用变得非常高效。通过 `timm.create_model()` 函数，可以快速实例化任何模型并选择是否加载预训练权重。模型的输入通道数、输出类别数等参数也支持灵活配置，方便用户自定义任务。除了模型本身， `timm` 还集成了多种实用工具，比如优化器、学习率调度器、训练脚本模板、混合精度训练支持、数据增强策略（如 RandAugment 和 AutoAugment）等，适合进行快速实验和大规模训练。该库对导出模型、迁移学习以及与 torchvision 数据加载器的结合使用也非常友好，并支持高效部署（如 ONNX 导出）。

注意，模型下载使用到了hugging face，请在执行的代码前面添加这个指令，保证你可以下载下来

```
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com
```

训练之前，大家需要设置一些基本信息，标着todo字样的都是大家需要进行修改的，这里我们以resnet50模型的训练为例。

 ![image-20250817114321633](https://i-blog.csdnimg.cn/img_convert/9f81f4cb0d63f01f5c9ad3e7cacd4c0b.png)

一开始执行之前会有一个会需要下载预训练模型到指定目录，由于众所周知的原因，大家需要提前先把模型下载下来放置到这个目录，这个大家自行探索。

 ![image-20221201114528829](https://i-blog.csdnimg.cn/img_convert/c9f0fde8a3c634f2ce546c8305489803.png)

右键直接运行 `train.py` 就可以开始训练模型，代码首先会输出模型的基本信息（模型有几个卷积层、池化层、全连接层构成）和运行的记录，如下图所示：

 ![image-20221128092952811](https://i-blog.csdnimg.cn/img_convert/113d1381035ae8ed6c067eb5df5c8c79.png)

运行结束之后，训练好的模型将会保存在一开始你指定的目录下，保存的逻辑是：如果当前轮的准确率比目前最好的准确率高就会保存。同时，模型训练过程中acc和loss的变化曲线也会保存在指定的保存目录下，如下所示。

 ![results](https://i-blog.csdnimg.cn/img_convert/ffbcd53149bb7842efe07e99398cffe8.png)

上面的图表示准确率的变化曲线，下面的图表示loss的变化曲线，其中蓝色的为为训练集，橘黄色的为验证集，从图中可以看出，随机训练过程的进行，模型在慢慢收敛直到稳定。

#### 模型的验证

`test.py` 是模型测试的主程序，在进行模型测试之前，你需要对标注了todo文字所在行的内容进行修改，如下图所示，并且你的测试一定是在训练之后进行的。

 ![image-20250817114439847](https://i-blog.csdnimg.cn/img_convert/c6c104976cb6ae60caef8dd277bddaf9.png)

修改完毕之后，右键执行 `test.py` 就可以对你指定的模型在测试集上进行测试，测试结束，将会输出模型在数据集上的F1、Recall和ACC等指标，并且会生成由每类分别的准确率构成的热力图保存在record目录下，如下图所示。

 ![](https://i-blog.csdnimg.cn/img_convert/ce5c10c018d9f856e0ea9ba06e1a4c9d.png)

#### 测试脚本的开发

为了方便大家对任意的图片进行预测，我这里封装了两个函数， `predict_batch` 函数负责对一个文件夹下的所有图片进行预测， `predict_single` 函数负责对单张图片进行预测。通过这两个函数进行扩展，你可以完成对视频的实时预测，或者是通过HTTP接口应用在你网站上的后端程序等，具体用途需要靠大家发挥想象力了。

模型预测的代码是 `predict.py` 文件，该文件中包含了两个函数，分别是文件夹预测和单张图片预测的函数，在实际运行的时候，你需要在main函数中指定具体要执行的是哪个函数。执行之前，需要设置一些基本参数，需要设置的参数我用todo进行了标记，大家按照标记进行修改就可以。

 ![image-20221128155931897](https://i-blog.csdnimg.cn/img_convert/87eeba415c9f930679925a2e1534cdeb.png)

##### 对文件夹下的图片进行预测

对文件夹的图片进行预测的时候，需要传入三个参数，分别是模型地址、需要预测的文件夹的地址、预测结果保存的地址（模型会按照预测结果放在对应的类别文件夹中）

比如对于下面的预测目录 `D:\upppppppppp\cls\cls_torch_tem\images\test_imgs\mini` ，我希望结果保存在目录 `D:\upppppppppp\cls\cls_torch_tem\images\test_imgs\mini_result` 下。

 ![image-20221128211734789](https://i-blog.csdnimg.cn/img_convert/e132c3eb0c21c57e24d284344763af1a.png)

打开mini_result文件夹，可以看到每个图片都按照预测结果放在了不同的文件夹下。

 ![image-20221128212610892](https://i-blog.csdnimg.cn/img_convert/6c491e99850e99c9c733826a34687435.png)

##### 对单张图片进行预测

比如我要对这样的一张图片进行预测，首先请标注这张图片的路径，后面将会使用到。

 ![image-20221128154426170](https://i-blog.csdnimg.cn/img_convert/25b0bc9716b3b9747138095d80657df2.png)

接着将图片路径写入 `predict_single` 函数，右键直接运行即可，运行效果如下图所示。

 ![image-20221128154733097](https://i-blog.csdnimg.cn/img_convert/27a2774f339ff3824d02e9ac9a51925e.png)

#### 图形化界面构建

图形化界面的构建依然通过pyqt5来编写，主要功能还是上传图片和对模型进行推理。

启动界面之前需要设置几个关键的参数，如下图所示：

 ![image-20221201113744210](https://i-blog.csdnimg.cn/img_convert/fb6c506a3df0674d87f1c601bc8550d2.png)

之前有小伙伴比较好奇上传和推理的逻辑在哪，这部分主要是通过pyqt中信号与槽的机制来进行实现的，比如对于推理按钮，我们首先生成一个button，然后通过clicked.connect和具体事件绑定起来即可实现调用模型推理的功能。

 ![image-20221201113948361](https://i-blog.csdnimg.cn/img_convert/65de65da5edb8934ee3e30a6cc4feeb6.png)

 ![image-20221201114048531](https://i-blog.csdnimg.cn/img_convert/ce429d357d0ad713b3365d304847d72a.png)

之后，直接启动window.py即可运行。

 ![image-20250817131808974](https://i-blog.csdnimg.cn/img_convert/fc52adebed4bec01a44a68d8a32480cb.png)

#### 模型其他信息查看

##### 模型结构查看

一般在论文中，为了让我们的网络看起来方便读一些，我们需要查看网络具体的模型结构，这个时候需要借助 [netron](https://netron.app/) 这个软件，在进行查看之后，需要将我们的模型转换为onnx格式的模型，onnx格式的模型也是后续实际在windows上做模型部署用的形式，这个我们会单独开一个章节来讲。

比如以我们训练好的resnet50模型为例，代码在‘工具类-onnx模型导出.py’，和之前一样，还是需要修改几个固定的参数之后直接右键执行即可，即可得到转换为了onnx格式的模型。

 ![image-20250613234736780](https://i-blog.csdnimg.cn/img_convert/0ae7956f880b399a270291cd8bf5c6b3.png)

然后将模型拖动到netron软件中即可查看具体的网络结构。

 ![image-20221201113334987](https://i-blog.csdnimg.cn/img_convert/63f17df5d8b5dff0ee9f58f173967d75.png)

##### 模型参数量查看

另外，有的时候我们需要查看模型的参数量和计算量，这里的计算量一般用flops来进行表示，但是实际上这里的计算量和fps不是严格意义上相关的，有的时候不一定的是flops低，模型的推理速度就快。模型实际的运行快慢还是需要看fps的，这部分的代码我没有写（一个简单的思路：就是算推理一个文件夹的图片花费了多少s，然后用图片数量除以时间， 就可以算出来fps）。

计算参数量的代码在 `get_flops.py` 文件，同样将todo的几个参数修改之后直接运行即可查看。

 ![image-20250613234806070](https://i-blog.csdnimg.cn/img_convert/d55ee697e21c91897c3325d2b822948a.png)

