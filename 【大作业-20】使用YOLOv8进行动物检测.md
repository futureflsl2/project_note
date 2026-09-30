# 【大作业-20】使用YOLOv8进行动物检测

## 【大作业-20】使用YOLOv8进行动物检测

Hello，这里是肆十二，今天我们继续来更新使用我们的大作业系列，我们之前录制了一期详细的关于yolov8的介绍，包含yolov8的原理介绍和代码解析。后续，我们将会围绕这个教程，针对一系列的数据集进行更新，主要是帮助各位没有显卡的小伙伴训练出你想要的模型。今天我们将会围绕动物检测这个主题展开，我们将会提供给大家训练好的yolov3、yolov5和yolov8的模型以及标注好的数据集，并且也会给出一个封装好的图形化的界面提供给大家进行修改。当然，为了恰饭，这部分训练好的模型和标注好的数据集是需要大家付费的，粉丝限时9.9元，或者你也可以选择自己使用labelimg来自行标注数据和训练，训练的方式和视频所示都是一致的，并且图形化的界面也是通用的，只需要你在代码中修改为自己的数据集即可。

另外，文档方面，后续我也会按照一个标准的文档进行输出，大家只需要稍作修改，就能摇身一变成为你自己的课程设计报告。



<div align="center" style="border: 3px solid gray;border-radius: 27px;overflow: hidden;"> <a class="link-info" href="https://player.bilibili.com/player.html?aid=113050787053936&autoplay=0" rel="nofollow" title="【大作业-20】用yolov8做动物检测">【大作业-20】用yolov8做动物检测</a><iframe id="4GKcTFg3-1758450647628" frameborder="0" src="https://player.bilibili.com/player.html?aid=113050787053936&amp;autoplay=0" allowfullscreen="true" data-mediaembed="bilibili" style="width: 100%; aspect-ratio: 2;" allow="fullscreen" loading="lazy"></iframe></div>



### 介绍

使用深度学习技术进行动物检测具有重要的背景和意义，以下是简要概述：

#### 背景：

1. **生态保护需求** ：随着全球生态系统的变化和人类活动的增加，许多野生动物的栖息地受到威胁。监测和保护野生动物成为了生物学家和环保组织的重要任务。

2. **传统方法的局限性** ：传统的动物检测方法通常依赖于人工观察和安装在野外的相机陷阱。这些方法不仅耗时耗力，而且容易受到环境因素的影响，数据的获取和分析也相对缓慢。

3. **技术进步** ：近年来，计算机视觉和深度学习技术的飞速发展，使得自动化图像分析成为可能。通过深度学习模型，计算机能够从大量图像中快速准确地识别和分类动物，大大提升了检测效率和准确率。

#### 意义：

1. **提升效率** ：深度学习技术可以自动处理和分析海量的野外监控图像，显著减少了人工参与的时间和成本，尤其是在大规模生态研究中。

2. **准确性高** ：相比传统方法，深度学习模型能够更精确地识别不同种类的动物，甚至可以在复杂环境下（如动物伪装或光照变化）保持较高的检测精度。

3. **数据积累与分析** ：通过自动化检测，大量的野外数据得以快速积累，这些数据可以用于进一步的生态研究、物种保护策略制定、以及环境变化的长期监测。

4. **保护濒危物种** ：深度学习技术的应用能够帮助研究人员快速发现并追踪濒危物种的活动，为物种保护提供关键数据支持。

5. **无人机和卫星应用** ：随着无人机和卫星技术的发展，深度学习技术还可以结合遥感数据，实现大范围内的动物群体监测，扩展了动物检测的空间和时间维度。

总体来说，使用深度学习技术进行动物检测不仅推动了生态学研究的进展，也为野生动物保护提供了强有力的技术支持。

### 相关工作

在过去五年中，深度学习在动物检测领域取得了显著进展。以下是一些关键研究工作及其对应的参考文献：

#### 1. **基于深度学习的野生动物检测与分类** 

- **研究内容** : 该研究提出了一种基于卷积神经网络（CNN）的模型，用于自动检测和分类野生动物图像。模型通过在大型数据集上训练，能够在不同环境下识别多种动物种类。

- **参考文献** : Norouzzadeh, M. S., Nguyen, A., Kosmala, M., Swanson, A., Palmer, M. S., Packer, C., & Clune, J. (2018). Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning. *Proceedings of the National Academy of Sciences* , 115(25), E5716-E5725. doi:10.1073/pnas.1719367115

#### 2. **利用无人机和深度学习检测海洋哺乳动物** 

- **研究内容** : 本研究开发了一种结合无人机图像和深度学习算法的方法，用于检测和跟踪海洋哺乳动物。该方法解决了传统手动分析海洋动物影像的效率低下问题。

- **参考文献** : Gray, P. C., Bierlich, K. C., Mantell, S. A., Friedlaender, A. S., & Johnston, D. W. (2019). Drones and convolutional neural networks facilitate automated and accurate cetacean species identification and photogrammetry. *Methods in Ecology and Evolution* , 10(9), 1490-1500. doi:10.1111/2041-210X.13246

#### 3. **深度学习在夜间动物检测中的应用** 

- **研究内容** : 该研究探索了如何利用红外摄像头捕捉的夜间图像，并通过深度学习算法来检测和识别野生动物。该技术对夜间监测有着重要意义，特别是在缺乏光照的条件下。

- **参考文献** : Tabak, M. A., Norouzzadeh, M. S., Wolfson, D. W., Sweeney, S. J., VerCauteren, K. C., Snow, N. P., & Beasley, J. C. (2019). Machine learning to classify animal species in camera trap images: applications in ecology. *Methods in Ecology and Evolution* , 10(4), 585-590. doi:10.1111/2041-210X.13120

#### 4. **使用深度学习进行水下动物检测** 

- **研究内容** : 该研究开发了一种新的深度学习框架，用于检测水下视频中的动物。该框架能够处理水下环境中的噪声和光线变化，从而提高检测的可靠性。

- **参考文献** : Villon, S., Mouillot, D., Chaumont, M., Darling, E. S., Subsol, G., Claverie, T., & Villéger, S. (2018). A deep learning method for accurate and fast identification of coral reef fishes in underwater images. *Ecological Informatics* , 48, 238-244. doi:10.1016/j.ecoinf.2018.09.002

#### 5. **利用深度学习模型检测濒危动物** 

- **研究内容** : 该研究专注于使用深度学习模型检测濒危物种，如大熊猫和雪豹。通过训练大规模的数据集，研究人员能够提高濒危物种的检测精度，支持生态保护工作。

- **参考文献** : Chen, X., Zhang, X., Wang, S., Zhang, J., Hu, J., & Liu, L. (2020). Deep learning-based classification of remote camera images for wild animal monitoring. *Remote Sensing* , 12(3), 482. doi:10.3390/rs12030482

这些研究展示了深度学习技术在动物检测领域的广泛应用，涵盖了从陆地到海洋、从白天到夜间的多种场景，推动了生态监测和动物保护的科学进步。

### 本文方法

如果正在看的小伙伴对原理的部分不感兴趣可以直接跳到后面的实操部分。

特别感谢MMYOLO官方提供了优秀的图解： [mmyolo/configs/yolov8/README.md at dev · open-mmlab/mmyolo (github.com)](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/README.md) 

Ultralytics开发的YOLOv8是一款尖端、最先进（SOTA）的模型，它借鉴了之前YOLO版本的成功经验，并引入了新的特性和改进，以进一步提高性能和灵活性。YOLOv8旨在实现快速、准确和易于使用，因此是各种目标检测、图像分割和图像分类任务的绝佳选择。注意，此时的YOLOv8的模型已经基本完成了最终的进化，除了支持最经典的目标检测任务之外，还添加了对语义分割、分类和追踪等任务的支持。当然我们本期还是选择大家最熟悉的检测任务来进行展开，关于后续的其他任务我们再另外录制。

 ![Ultralytics YOLO supported tasks](https://i-blog.csdnimg.cn/img_convert/5bddb941fa9cda53301d53e81ee1bec1.png)

首先我们来看一下YOLOv8算法的性能。下图是官方提供了性能图，其中左图的横坐标表示的是网络的参数量，右图的横坐标表示的网络在A100显卡上的推理速度，纵坐标方面表示表示的都是模型的精度。可以看出，YOLOv8模型的在同样的参数量下，比其他系列的YOLO模型有明显的精度提升，在右图展示的同样的map精度下，YOLOv8的模型也同样有更快的速度，还真就是那个更高、更快、更强。

 ![Ultralytics YOLOv8](https://i-blog.csdnimg.cn/img_convert/0074ed1c7084476f41a9c63377936a5e.png)

下面的表格则是来自 [YOLOv8 - Ultralytics YOLO Docs](https://docs.ultralytics.com/models/yolov8/#overview) 提供的在coco数据集上的测试结果，从表中可以看出，对于他的nano模型而言，在只有3.2M的参数量下，就可以达到37.3的mAP，非常优秀的数值表现。

 ![image-20240816165549221](https://i-blog.csdnimg.cn/img_convert/3718197023a4cd997133bec49d515cc1.png)

YOLOv8算法的核心点可以总结为下面几点。

1. **给出了一个全新的视觉模型** ，保持精度的同时，实现了较高的检测速度，并且同时支持支持图像分类、物体检测和实例分割等多种视觉任务。并且提供了多个规模的模型（nano、small、medium、large和x-large），满足用户不同场景的需要。

2. **新的骨干网络** ：YOLOv8引入了一个新的骨干网络，可能参考了YOLOv7 ELAN设计思想，将YOLOv5中的C3结构换成了梯度流更丰富的C2f结构，并对不同尺度模型调整了不同的通道数，大幅提升了模型性能。

3. **解耦头的设计** ：YOLOv8的Head部分从原先的耦合头变成了解耦头结构，将分类和检测头分离，并且从Anchor-Based转变为Anchor-Free，简化了模型结构并提高了推理速度。

4. **新的损失函数** ：YOLOv8在Loss计算方面采用了TaskAlignedAssigner正样本分配策略，并引入了Distribution Focal Loss，确保了检测结果的准确性和鲁棒性。

OK，说完这里的性能表现，我们就一起来看看YOLOv8结构方面的内容吧。

#### 结构说明

首先是YOLOv8的网络结构图

 ![222869864-1955f054-aa6d-4a80-aed3-92f30af28849](https://i-blog.csdnimg.cn/img_convert/cb85dd772cf421271a17b0fe419f36b6.jpeg)

- 骨干网络部分：

  骨干网络部分的c2f结构可能借鉴了 [YOLOv7](https://github.com/WongKinYiu/yolov7) 的设计。将原先的c3模块更新了c2f的模块，其中c3表示使用了3个常规的卷积模块，c2f表示使用了2个卷积模块并且更快（fast）。在不改变原始架构和梯度传输路径的前提下， 使用分组卷积踢来以及使用洗牌和合并的操作组合不同组的特征，增强模型从不同特征图中的学习能力，达到改善参数的作用。

  下图是YOLOv7中原文提到的Elan的结构，主要是使用了更多的连接和合并的操作。

   ![image-20240816171847228](https://i-blog.csdnimg.cn/img_convert/97ec2262299e1262eadecdb3a41d5f9c.png)

  下面的图是YOLO模型原先使用的C3结构和新的C2f的结构。

   ![module](https://i-blog.csdnimg.cn/img_convert/4fc5e93b0dac2b87d1603e8dcf746982.png)

  和左图相比，增加了更多的跳层的连接和split操作。下面是两个部分具体实现的源码。从源码中可以看出，对于原先的结构而言，只使用了一次cat操作，对于c2f结构而言，使用了多次的split操作和cat操作，按照原文的说法是增加了梯度流。

  ```python
  class C3(nn.Module):
      """
      这里是使用了3个卷积层的csp结构
      CSP Bottleneck with 3 convolutions.
      """

      def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
          """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
          super().__init__()
          c_ = int(c2 * e)  # hidden channels
          self.cv1 = Conv(c1, c_, 1, 1)
          self.cv2 = Conv(c1, c_, 1, 1)
          self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
          self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

      def forward(self, x):
          """Forward pass through the CSP bottleneck with 2 convolutions."""
          return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
  class C2f(nn.Module):
      """
      这里使用了分支处理的操作，使用的是通过关闭残差链接的方式实现
      先进行分支的操作然后再进行特征融合的操作
      Faster Implementation of CSP Bottleneck with 2 convolutions.
      """

      def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
          """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
          expansion.
          """
          super().__init__()
          self.c = int(c2 * e)  # hidden channels
          self.cv1 = Conv(c1, 2 * self.c, 1, 1)
          self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
          self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

      def forward(self, x):
          """Forward pass through C2f layer."""
          y = list(self.cv1(x).chunk(2, 1))
          y.extend(m(y[-1]) for m in self.m)
          return self.cv2(torch.cat(y, 1))

      def forward_split(self, x):
          """Forward pass using split() instead of chunk()."""
          y = list(self.cv1(x).split((self.c, self.c), 1))
          y.extend(m(y[-1]) for m in self.m)
          return self.cv2(torch.cat(y, 1))
  ```

  除此之外，YOLOv8的主干还去掉了neck部分中的2个卷积层，以及将block的数量从原先的 3-6-9-3 改成了 3-6-6-3。另外，在之前的YOLOv5的网络结构中，只需要更改一下w和h系数就能统一适配不同规模的模型，但是对于YOLOv8而言，其中N和S的结构一致，L和X的结构一致，这两对模型可以只通过修改缩放系数就完成替换。在YOLOv10中，作者也提到了这个观点，为了追求网络的灵活性，导致网络同质化比较严重，其中有些冗余的模块是可以去除的，也说明现在的网络结构向着比较定制化的方向进行，当然，这句话是我的个人观点。

- 解码头部分：

  解码头的部分选择使用了分类的解码头，也就是边界框回归是一个分支以及分类的是一个分支。如下图所示，上面的子图是原先的解码头部，经过主干网络进行特征提取之后得到特征图，之后直接进入一个模块中进行解码，这里的数值计算包含3个部分，分别是用于边界框回归的CIoU、用于置信度计算的obj和用于分类类别计算的CLS。改进之后的头部如下面的子图所示，经过主干网络进行特征提取之后，上面的子分支用于回归，下面的子分支则用于分类，去除了之前的obj的部分，在回归的分支中，使用的是Distribution Focal Loss。

   ![head](https://i-blog.csdnimg.cn/img_convert/d98b8d4c1c9e5f99d1adcc84ee011768.png)

  其中DFL损失函数的定义如下，通俗来讲就是训练的过程中，目标的边界框不应该是一个确定的数值，目标的边界框应该是一个分布，比如对于浪花这个物体而言，他的边界就是不清晰的，通过这样的损失函数可以减少网络在训练过程中出现的过拟合的现象。

   ![image-20240816235150126](https://i-blog.csdnimg.cn/img_convert/e7f6b8c9935f62b70f4ca6bb83578158.png)

  其中，DFL实现的代码如下：

  ```python
  def distribution_focal_loss(pred, label):
      r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
      Qualified and Distributed Bounding Boxes for Dense Object Detection
      <https://arxiv.org/abs/2006.04388>`_.

      Args:
          pred (torch.Tensor): Predicted general distribution of bounding boxes
              (before softmax) with shape (N, n+1), n is the max value of the
              integral set `{0, ..., n}` in paper.
          label (torch.Tensor): Target distance label for bounding boxes with
              shape (N,).

      Returns:
          torch.Tensor: Loss tensor with shape (N,).
      """
      #label为y, pred为y^(y的估计值）
      #因为y_i <= y <= y_i+1(paper)
      #取dis_left = y_i, dis_right = y_i+1
      dis_left = label.long()
      dis_right = dis_left + 1
      weight_left = dis_right.float() - label #y_i+1-y
      weight_right = label - dis_left.float() #y-y_i
      #paper中的log(S)这里用CE
      loss = (
          F.cross_entropy(pred, dis_left, reduction="none") * weight_left
          + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
      )
      return loss
  ```

#### 损失函数说明

YOLOv8的loss计算包含了分类和回归两个部分，没有了之前的objectness的分支部分，其中分类的分支采用的是BCE Loss，回归的分支使用了两个部分，分别是上面提到的Distribution Focal Loss和CIoU Loss，3个损失函数按照一定的权重比例进行加权。

关于正负样本的分配，其中YOLOv5中使用的是静态的分布策略，简单来说，静态的分布策略是将标签中的GT Box和Anchor Templates模板计算IoU，如果IoU大于设定的阈值就认为是匹配成功，匹配成功的边界框将会参与到CIoU Loss的计算中。当然这里所述的是简化的版本，实际子计算的过程中还会去计算GT Box和对应的的Anchor Templates模板高宽的比例。假设对某个GT Box而言，其实只要GT Box满足在某个Anchor Template宽和高的× 0.25 0.25倍和4.0倍之间就算匹配成功。关于这部分更详细的解释可以看 [YOLOv5网络详解_yolov5网络结构详解-CSDN博客](https://blog.csdn.net/qq_37541097/article/details/123594351?spm=1001.2014.3001.5501) 。在YOLOv8中，使用的是动态分布的策略（YOLOX 的 simOTA、TOOD 的 TaskAlignedAssigner 和 RTMDet 的 DynamicSoftLabelAssigner），这里直接使用的是 TOOD 的 TaskAlignedAssigner。 TaskAlignedAssigner 的匹配策略简单总结为： 根据分类与回归的分数加权的分数选择正样本。

 ![image-20240817001553493](https://i-blog.csdnimg.cn/img_convert/fb0dd17a468981a4afb9874b4bd7c6ff.png)

s是标注类别对应的预测分数值，u是预测框和gt框之间的iou。计算出分数之后，根据分数选取topK大的作为正样本，其余作为负样本。

#### 数据增强说明

数据增强的部分和YOLOv5基本保持了一致，包含了颜色变换、马赛克数据增强、随机剪切等一系列常规的数据增强的方式。并且使用YOLOX的数据增强策略，在前面的部分使用数据增强，而在最后的10个epoch中关闭数据增强。如下图所示。

 ![head](https://i-blog.csdnimg.cn/img_convert/c21701a0917f44fa74ccb04eea396f86.png)

对于一些常见的数据增强的方式的说明。

 ![image-20240817002209214](https://i-blog.csdnimg.cn/img_convert/a11168026e1dc47f708c96a7fdb9c1fa.png)

#### 训练策略说明

YOLOv8 的推理过程和 YOLOv5 几乎一样，唯一差别在于前面需要对 Distribution Focal Loss 中的积分表示 bbox 形式进行解码，变成常规的 4 维度 bbox，后续计算过程就和 YOLOv5 一样了。

以 COCO 80 类为例，假设输入图片大小为 640x640，MMYOLO 中实现的推理过程示意图如下所示：

其推理和后处理过程为：

 ![head](https://i-blog.csdnimg.cn/img_convert/ecdaf85f27c96aa046e342f00ae4d392.png)

**(1) bbox 积分形式转换为 4d bbox 格式** 

对 Head 输出的 bbox 分支进行转换，利用 Softmax 和 Conv 计算将积分形式转换为 4 维 bbox 格式

**(2) 维度变换** 

YOLOv8 输出特征图尺度为 `80x80` 、 `40x40` 和 `20x20` 的三个特征图。Head 部分输出分类和回归共 6 个尺度的特征图。 将 3 个不同尺度的类别预测分支、bbox 预测分支进行拼接，并进行维度变换。为了后续方便处理，会将原先的通道维度置换到最后，类别预测分支 和 bbox 预测分支 shape 分别为 (b, 80x80+40x40+20x20, 80)=(b,8400,80)，(b,8400,4)。

**(3) 解码还原到原图尺度** 

分类预测分支进行 Sigmoid 计算，而 bbox 预测分支需要进行解码，还原为真实的原图解码后 xyxy 格式。

**(4) 阈值过滤** 

遍历 batch 中的每张图，采用 `score_thr` 进行阈值过滤。在这过程中还需要考虑 **multi_label 和 nms_pre，确保过滤后的检测框数目不会多于 nms_pre。** 

**(5) 还原到原图尺度和 nms** 

基于前处理过程，将剩下的检测框还原到网络输出前的原图尺度，然后进行 nms 即可。最终输出的检测框不能多于 **max_per_img。** 

有一个特别注意的点： **YOLOv5 中采用的 Batch shape 推理策略，在 YOLOv8 推理中暂时没有开启，不清楚后面是否会开启，在 MMYOLO 中快速测试了下，如果开启 Batch shape 会涨大概 0.1~0.2。** 

### 实验结果

实验结果的部分，多为看图写话，这里我简单放置一些图像和说明，大家自行进行扩展即可。

#### 实验设置

**硬件参数** :

显卡方面为NVIDIA RTX 3060：

 ![image-20240822120004346](https://i-blog.csdnimg.cn/img_convert/94631ffed013b193a5e40031a45234e6.png)

其余硬件和系统参数如下：

 ![image-20240822115900542](https://i-blog.csdnimg.cn/img_convert/2c45ddf442e8fabee6ecc48e1093f041.png)

**训练参数** :

- **学习率** : 初始学习率设置为0.001，使用余弦退火或指数衰减策略逐步降低学习率。

- **批量大小** : 16至32，具体取决于显存容量和硬件性能。

- **优化器** : 使用AdamW或SGD优化器，权重衰减设置为0.0005，以控制模型复杂度。

- **损失函数** : 使用YOLOv8默认的损失函数（如CIoU Loss和DFL Loss），结合分类损失和定位损失进行优化。

- **训练轮数** : 50至100个epoch，视模型的收敛情况而定。

**测试与评估** :

- **评估指标** : 使用mAP（mean Average Precision）作为主要评估指标，阈值设置为0.5 (mAP@0.5) 和0.5:0.95 (mAP@0.5:0.95) 以全面评估模型性能。

- **实时性测试** : 在测试集上评估模型的FPS（Frames Per Second）表现，以确保模型满足实时检测的要求。

- **跨域测试** : 使用来自不同场景的数据集（如夜间监控、工地监控）进行测试，评估模型的泛化能力。

#### 数据集介绍

数据集方面我们搜集了大概有1w个动物实例的数据集，数据集类别的整体分布如下，包含3个类别，分别是行人、头以及动物，这里的数据集大家也可以自行进行补充应用在骑车的动物检测或者是其他各种涉及到动物检测的场景中。

 ![image-20240830153921902](https://i-blog.csdnimg.cn/img_convert/6d7bdd43286ee498d53bf1a2d8e82926.png)

这里有一些部分数据集的示例，大家在自己的课程设计报告中也可以进行放置。

 ![val_batch2_pred](https://i-blog.csdnimg.cn/img_convert/8dff444ceae4919f8b1130f4ef741d06.jpeg)

#### 实验的配置和实验结果

实验的配置部分还是和之前一样。

配置之前请务必进行国内源的切换，这样可以帮助你进行包安装的加速。

```bash
conda config --remove-key channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

- 建立虚拟环境

  ```bash
  conda create -n yolov8
  ```

- 激活虚拟环境并安装pytorch

  ```bash
  conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
  conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令
  conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
  ```

- 安装其他依赖库并使用pycharm打开

  ```bash
  pip install -v -e .
  ```

下面是一些实验结果的展示。

首先是训练的结果曲线方面：

**Loss曲线分析：** 

- **train/box_loss、train/cls_loss、train/dfl_loss** ：这些曲线展示了模型在训练过程中不同类型损失函数的变化情况。总体来看，三个损失曲线均呈现出逐渐下降的趋势，这表明模型在训练过程中逐渐学会了更好地拟合训练数据。值得注意的是，曲线在接近第100个epoch时出现了一个突出的上升，这是因为在训练过程中可能进行了学习率调整或者其他训练策略的改变，导致损失短暂增高。

- **val/box_loss、val/cls_loss、val/dfl_loss** ：验证集上的损失曲线与训练集上相似，也呈现出下降趋势，表明模型在验证数据上也有良好的表现。验证损失的下降是模型有效学习的一个指示。这里同样在第100个epoch左右也出现了小幅上升，可能是因为相同的训练策略调整。

**Precision和Recall曲线分析：** 

- **metrics/precision(B)**和**metrics/recall(B)** ：这些曲线显示了模型在训练过程中精确度（Precision）和召回率（Recall）的变化。随着训练的进行，这两个指标都在逐步提升，表明模型的检测效果在持续改善。

**mAP曲线分析：** 

- **metrics/mAP50(B)**和**metrics/mAP50-95(B)** ：这些曲线展示了平均精度（mAP）的变化。mAP是衡量目标检测模型整体性能的关键指标。曲线显示出mAP值在训练过程中逐渐提升，说明模型的检测性能在不断提高。

**平滑曲线和原始曲线的对比：** 

- 图中有两种曲线：实线代表模型的实际训练结果，虚线则是经过平滑处理后的结果。平滑曲线可以帮助我们更清楚地看到趋势，而不是被短期波动干扰。整体趋势在两种曲线中是一致的，平滑曲线帮助我们更清楚地看到损失下降和mAP提升的过程。

**学习率调整或其他训练策略的影响：** 

- 在多个曲线图中，第100个epoch附近的突变可能表明了学习率调整或其他训练策略（如warm-up、冷启动等）的影响，这是常见的训练手法，在大部分情况下，这些调整有助于模型突破某些瓶颈。

 ![results](https://i-blog.csdnimg.cn/img_convert/43f59b16e1ad55e8d88076264158815f.png)

这张图展示了YOLOv8模型在不同类别上的Precision-Recall曲线，帮助我们评估模型在各个类别上的分类性能。以下是对该图的详细分析，按照论文中的标准结构进行描述：

1. 图像描述：
   图像展示了多个类别的Precision-Recall曲线以及所有类别的总体表现。每条曲线表示该类别的Precision与Recall之间的关系，曲线越靠近右上角，表示模型在该类别上有更好的检测性能。粗线代表所有类别的平均Precision-Recall曲线（mAP@0.5）。

2. 整体性能分析：
   从图中可以看出，模型的总体mAP@0.5（粗蓝线）为0.687，表明在所有类别上，模型在检测任务中的平均性能较为良好。

3. 各类别性能对比：
   最高表现类别：马（horse）类别的表现最佳，其mAP@0.5为0.796。这表明模型在马类物体的检测中精度和召回率都较高。
   次高表现类别：猫（cat）类别紧随其后，mAP@0.5为0.768，表明模型在此类物体上也具有良好的检测效果。
   最低表现类别：牛（cow）类别的表现最差，mAP@0.5仅为0.568，说明模型在检测此类物体时容易出现误检或漏检。

4. Precision-Recall曲线形状分析：
   较陡峭的曲线：例如马（horse）和猫（cat）的曲线，表明模型在这些类别上能够在高Recall值下仍保持较高的Precision，这对于检测任务来说是非常理想的。
   较平缓的曲线：例如牛（cow）类别的曲线，表现出随着Recall的增加，Precision迅速下降，表明模型在检测该类物体时可能存在较高的误检率。

5. 多类别检测的挑战：
   图中可以看出，不同类别的PR曲线形态差异较大，这可能是由于类别间的样本不均衡或模型在某些类别上的特征学习不足导致的。这些差异性表明在多类别检测任务中，不同类别的检测难度不同。

6. 改进建议：
   针对表现较差的类别（如牛），可以考虑通过数据增强、类别平衡或者调整损失函数权重等方法来提升模型在这些类别上的检测性能。
   进一步的模型优化可以针对不同类别分别进行，以提升整体模型的mAP表现。

7. 总结：
   本图通过Precision-Recall曲线有效展示了YOLOv8模型在多类别检测任务中的性能差异。尽管总体表现良好，但某些类别如牛的检测性能仍有提升空间。针对这些差异，未来的工作可以针对性地进行模型优化和数据处理。

 ![PR_curve](https://i-blog.csdnimg.cn/img_convert/6bb9d258fb16b09d658051a97abf9b5a.png)

这里的图示可以借助一些AI模型来帮助你进行解读，如下所示。

 ![image-20240826151927945](https://i-blog.csdnimg.cn/img_convert/1ab45d04e31522e664baeea74b2b4722.png)

### 结论

在本实验中，我们使用YOLOv8模型进行了动物检测任务的训练。实验的主要目标是构建一个能够精确识别多种动物类别（如鸟、猫、狗、牛、马、羊）的目标检测模型，并评估其在各个类别上的性能。通过一系列训练和评估过程，我们得到了以下结论：

1. **模型性能** ：

   - 模型总体表现良好，mAP@0.5为0.687，表明在所有类别上，模型具有较为稳定的检测能力。

   - 在不同类别上的检测性能存在较大差异，表现最好的类别为马（mAP@0.5为0.796），而表现最差的类别为牛（mAP@0.5为0.568）。

2. **Precision-Recall分析** ：

   - Precision-Recall曲线显示出模型在大多数类别上能够较好地平衡精确度和召回率，尤其是在马和猫类物体的检测中表现突出。

   - 对于牛类物体，Precision-Recall曲线显示出随着召回率的增加，精确度显著下降，暗示模型在该类别上的误检率较高。

3. **训练过程** ：

   - 损失曲线（如box loss, cls loss, dfl loss）在训练和验证集上均表现出良好的收敛性，但在某些阶段（如第100个epoch附近）出现了短暂的波动，这可能是由于学习率调整或其他训练策略导致的。

   - mAP曲线随着训练过程的进行逐步提升，说明模型在不断学习并优化其对不同类别的检测能力。

### 未来工作

为了进一步提升模型性能，特别是针对在某些类别上的低检测效果，未来的工作可以考虑以下几个方向：

1. **数据增强与类别平衡** ：

   - 增加数据集中的样本数量，特别是对于表现较差的类别，如牛类，可以通过数据增强（如旋转、缩放、色彩变换等）来增加样本的多样性，从而提高模型的泛化能力。

   - 在训练过程中对类别进行平衡，确保模型在所有类别上都有足够的学习机会，这可以通过调整损失函数中的类别权重来实现。

2. **模型优化** ：

   - 考虑使用更复杂的模型结构或在现有YOLOv8基础上进行改进，以提高模型在难检测类别上的表现。例如，可以引入注意力机制或多尺度特征融合来增强模型对细小目标或复杂场景的检测能力。

   - 针对表现不佳的类别进行特定的损失函数设计，或者引入类别特定的学习率调整策略，以改善模型在这些类别上的表现。

3. **迁移学习与领域适应** ：

   - 应用迁移学习，将在更大规模或更多样化的数据集上预训练的模型进行微调，可能有助于提升当前模型在特定类别上的表现。

   - 研究领域适应技术，将模型从训练数据集迁移到更加多样化的实际应用场景中，从而提高模型的鲁棒性和实用性。

4. **进一步的评估与优化** ：

   - 在更多实际场景中对模型进行评估，以检测其在不同环境、光照条件、背景复杂度下的表现。根据评估结果进行针对性优化，确保模型在现实世界应用中的有效性。

   - 考虑对模型进行多次训练和交叉验证，以减少偶然性因素的影响，并确保结果的稳健性和可重复性。

通过这些改进和深入研究，预期可以进一步提升YOLOv8模型在动物检测任务中的表现，并将其应用于更广泛的实际场景，如野生动物监测、农场动物管理和智能视频监控等领域。