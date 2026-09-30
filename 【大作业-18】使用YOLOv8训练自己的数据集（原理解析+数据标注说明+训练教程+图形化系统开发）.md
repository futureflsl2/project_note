# 【大作业-18】使用YOLOv8训练自己的数据集（原理解析+数据标注说明+训练教程+图形化系统开发）

## 使用YOLOv8训练自己的数据集

Hello，大家好，本次我们来教大家使用YOLOV8训练自己的数据集。

> 视频地址： [手把手教你用YOLOv8训练自己的数据集（原理解析+代码实践）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1KHp2eREFZ/?spm_id_from=333.999.0.0&vd_source=2f9a4e63109c3db3be5e8078e5111776) 
> 
> CSDN资源地址： [使用YOLOv8训练自己的数据集（原理解析+数据标注说明+训练教程+图形化系统开发）-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/141301902) 

YOLO系列目前已经更新到了V10，并且YOLO系列模型已经目前稳定运行了一段时间。经过一段时间的准备，我们选择在暑期的这个时间点更新YOLOV8模型的教程，从原理、数据标注和环境配置一一展开讲解，帮助小伙伴们掌握YOLOv8的基本内容。注意本次的教程除了支持v8模型的训练，还适用v3、v5、v9、v10等一系列模型的训练。

为了帮助大家能灵活选择自己喜欢的内容，我们选择分P的方式进行更新。比如，有的小伙伴只喜欢理论的部分，有的小伙伴只喜欢实操的部分，这样大家可以根据自己的需要各取所需。

 ![image-20240818153803760](https://i-blog.csdnimg.cn/blog_migrate/5d0e99275e7f292a39b00e37d35bde6b.png)

本期内容可以通过三连的方式获取（三连之后在评论区中留下您的邮箱即可），也可以通过B站工房的方式或者csdn这篇博客上方的资源链接来进行获取。

### YOLOv8原理解析

如果正在看的小伙伴对原理的部分不感兴趣可以直接跳到后面的实操部分。

特别感谢MMYOLO官方提供了优秀的图解： [mmyolo/configs/yolov8/README.md at dev · open-mmlab/mmyolo (github.com)](https://github.com/open-mmlab/mmyolo/blob/dev/configs/yolov8/README.md) 

Ultralytics开发的YOLOv8是一款尖端、最先进（SOTA）的模型，它借鉴了之前YOLO版本的成功经验，并引入了新的特性和改进，以进一步提高性能和灵活性。YOLOv8旨在实现快速、准确和易于使用，因此是各种目标检测、图像分割和图像分类任务的绝佳选择。注意，此时的YOLOv8的模型已经基本完成了最终的进化，除了支持最经典的目标检测任务之外，还添加了对语义分割、分类和追踪等任务的支持。当然我们本期还是选择大家最熟悉的检测任务来进行展开，关于后续的其他任务我们再另外录制。

 ![Ultralytics YOLO supported tasks](https://i-blog.csdnimg.cn/blog_migrate/dd63b942078b1adb77bef945e43218a8.png)

首先我们来看一下YOLOv8算法的性能。下图是官方提供了性能图，其中左图的横坐标表示的是网络的参数量，右图的横坐标表示的网络在A100显卡上的推理速度，纵坐标方面表示表示的都是模型的精度。可以看出，YOLOv8模型的在同样的参数量下，比其他系列的YOLO模型有明显的精度提升，在右图展示的同样的map精度下，YOLOv8的模型也同样有更快的速度，还真就是那个更高、更快、更强。

 ![Ultralytics YOLOv8](https://i-blog.csdnimg.cn/blog_migrate/bc15b1cf737d5c52cedc59d5403128b5.png)

下面的表格则是来自 [YOLOv8 - Ultralytics YOLO Docs](https://docs.ultralytics.com/models/yolov8/#overview) 提供的在coco数据集上的测试结果，从表中可以看出，对于他的nano模型而言，在只有3.2M的参数量下，就可以达到37.3的mAP，非常优秀的数值表现。

 ![image-20240816165549221](https://i-blog.csdnimg.cn/blog_migrate/84bdd646c1e40ba856193b6b50b4918c.png)

YOLOv8算法的核心点可以总结为下面几点。

1. **给出了一个全新的视觉模型** ，保持精度的同时，实现了较高的检测速度，并且同时支持支持图像分类、物体检测和实例分割等多种视觉任务。并且提供了多个规模的模型（nano、small、medium、large和x-large），满足用户不同场景的需要。

2. **新的骨干网络** ：YOLOv8引入了一个新的骨干网络，可能参考了YOLOv7 ELAN设计思想，将YOLOv5中的C3结构换成了梯度流更丰富的C2f结构，并对不同尺度模型调整了不同的通道数，大幅提升了模型性能。

3. **解耦头的设计** ：YOLOv8的Head部分从原先的耦合头变成了解耦头结构，将分类和检测头分离，并且从Anchor-Based转变为Anchor-Free，简化了模型结构并提高了推理速度。

4. **新的损失函数** ：YOLOv8在Loss计算方面采用了TaskAlignedAssigner正样本分配策略，并引入了Distribution Focal Loss，确保了检测结果的准确性和鲁棒性。

OK，说完这里的性能表现，我们就一起来看看YOLOv8结构方面的内容吧。

#### 结构说明

首先是YOLOv8的网络结构图

 ![222869864-1955f054-aa6d-4a80-aed3-92f30af28849](https://i-blog.csdnimg.cn/blog_migrate/0e919b97e4b64cfc32bf174c037164cc.jpeg)

- 骨干网络部分：

  骨干网络部分的c2f结构可能借鉴了 [YOLOv7](https://github.com/WongKinYiu/yolov7) 的设计。将原先的c3模块更新了c2f的模块，其中c3表示使用了3个常规的卷积模块，c2f表示使用了2个卷积模块并且更快（fast）。在不改变原始架构和梯度传输路径的前提下， 使用分组卷积踢来以及使用洗牌和合并的操作组合不同组的特征，增强模型从不同特征图中的学习能力，达到改善参数的作用。

  下图是YOLOv7中原文提到的Elan的结构，主要是使用了更多的连接和合并的操作。

   ![image-20240816171847228](https://i-blog.csdnimg.cn/blog_migrate/c27ed347dcd6c450d4ea97d76bf6bc38.png)

  下面的图是YOLO模型原先使用的C3结构和新的C2f的结构。

   ![module](https://i-blog.csdnimg.cn/blog_migrate/8cfd42013526828e7cf7a9702ce2a598.png)

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

   ![head](https://i-blog.csdnimg.cn/blog_migrate/721a0c551fd978d3910d7960a77fa188.png)

  其中DFL损失函数的定义如下，通俗来讲就是训练的过程中，目标的边界框不应该是一个确定的数值，目标的边界框应该是一个分布，比如对于浪花这个物体而言，他的边界就是不清晰的，通过这样的损失函数可以减少网络在训练过程中出现的过拟合的现象。

   ![image-20240816235150126](https://i-blog.csdnimg.cn/blog_migrate/eca6703ecfc4e6f35ea23584c984cb5e.png)

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

 ![image-20240817001553493](https://i-blog.csdnimg.cn/blog_migrate/a74e8731777476c1b86a1750d1b05553.png)

s是标注类别对应的预测分数值，u是预测框和gt框之间的iou。计算出分数之后，根据分数选取topK大的作为正样本，其余作为负样本。

#### 数据增强说明

数据增强的部分和YOLOv5基本保持了一致，包含了颜色变换、马赛克数据增强、随机剪切等一系列常规的数据增强的方式。并且使用YOLOX的数据增强策略，在前面的部分使用数据增强，而在最后的10个epoch中关闭数据增强。如下图所示。

 ![head](https://i-blog.csdnimg.cn/blog_migrate/71f91ae058e6ff1f18192733903f40b6.png)

对于一些常见的数据增强的方式的说明。

 ![image-20240817002209214](https://i-blog.csdnimg.cn/blog_migrate/981cb3b7c53cf508ccde5009a7e2b269.png)

#### 训练策略说明

YOLOv8 的推理过程和 YOLOv5 几乎一样，唯一差别在于前面需要对 Distribution Focal Loss 中的积分表示 bbox 形式进行解码，变成常规的 4 维度 bbox，后续计算过程就和 YOLOv5 一样了。

以 COCO 80 类为例，假设输入图片大小为 640x640，MMYOLO 中实现的推理过程示意图如下所示：

其推理和后处理过程为：

 ![head](https://i-blog.csdnimg.cn/blog_migrate/1fd11a215281a4e1c2bb19709db7b2b8.png)

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

### 代码解析

下载代码之后，你将会看到下面的代码目录结构，其中 `42_demo` 是我准备的简易的执行文件，其余文件都是官方的文件和目录，每个文件大概的作用如下。

 ![image-20240818114837691](https://i-blog.csdnimg.cn/blog_migrate/82561248499db54edbe759fb94926517.png)

- 42_demo/

  这个目录下的文件是用于我们本次教程的脚本，我们将训练、测试、预测等脚本进行了单独的封装，方便初学者或者不是计算机专业的同学运行，每个脚本对应的含义如下。

   ![image-20240818121142997](https://i-blog.csdnimg.cn/blog_migrate/385ef38fa03a5309f427299e9a850951.png)

  其中比较重要的是训练的脚本 `start_train.py` ，这个脚本记录了数据的加载和一些训练的超参数，内容如下。

  ```python
  import time
  from ultralytics import YOLO

  # yolov8n模型训练：训练模型的数据为'A_my_data.yaml'，轮数为100，图片大小为640，设备为本地的GPU显卡，关闭多线程的加载，图像加载的批次大小为4，开启图片缓存
  model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
  results = model.train(data='A_my_data.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=4, cache=True)  # 开始训练
  time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用


  ```

  预测的测试脚本主要用于单张图像的检测，脚本为 `start_single_detect.py` 

  ```python
  from ultralytics import YOLO

  # Load a model
  model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

  # Run batched inference on a list of images
  results = model(["images/resources/demo.jpg", ])  # return a list of Results objects

  # Process results list
  for result in results:
      boxes = result.boxes  # Boxes object for bounding box outputs
      masks = result.masks  # Masks object for segmentation masks outputs
      keypoints = result.keypoints  # Keypoints object for pose outputs
      probs = result.probs  # Probs object for classification outputs
      obb = result.obb  # Oriented boxes object for OBB outputs
      result.show()  # display to screen
      result.save(filename="images/resources/result.jpg")  # save to disk

  ```

- docker/

  Docker 是一个开源的应用容器引擎，它允许开发者打包他们的应用以及依赖包到一个可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。容器是完全使用沙箱机制，相互之间不会有任何接口。

  简单来说，Docker 提供了一个可以运行你的应用程序的虚拟环境，但它比传统的虚拟机更加轻量和快速。使用 Docker，你可以轻松地在不同的机器和平台上部署、运行和管理应用程序，而无需担心环境配置和依赖问题。

  这个目录是根据用户不同的软硬件情况写的配置文件，但是一般情况下大家使用的不是很多，对于看我内容的小伙伴来说，大部分都是学生，使用的更少，所以这里的内容我们就不详细说明了。

   ![image-20240818123305515](https://i-blog.csdnimg.cn/blog_migrate/cf8914a053eba04b31806bc36af1c3b7.png)

- docs/

  这个目录用于放置对这个代码解释的官方文档，包含了各个不同的语言。

   ![image-20240818123410826](https://i-blog.csdnimg.cn/blog_migrate/40302a18f405f94a05fbe16af0caaee1.png)

- examples/

  这个目录下有官方提供的一些案例，并且包含了一些模型导出之后C++的调用脚本，这里的脚本大多数时候只有在实际部署的时候才会使用到，关于硬件部署是一个比较复杂的内容，这块的内容我们会单独抽时间来讲。

   ![image-20240818123441605](https://i-blog.csdnimg.cn/blog_migrate/a146ce25e507e6945d648d9b8e0c4712.png)

- test/

  test目录存放了一些自动化测试的脚本。

- ultralytics/

  该目录是整个项目的核心目录，存放了网络结构的底层实现和网络、数据集一系列的配置文件，平常修改网络结构和新增数据都会在这个目录下执行。

  assets目录下存放了两张经典的测试图像，用于模型的初步验证。

  cfg目录下存放了数据、模型和追踪的配置文件，举个例子，其中datasets下面的A_my_data.yaml就是我们本次教程使用的数据集配置文件，在这个路径中我们指明了数据集的路径和类别等信息。而models目录下的yolov8.yaml配置文件则指定了我们要使用到的模型。详细的内容如下：

  A_my_data.yaml

  ```python
  # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
  path: ../../person_42_yolo_format
  train: # train images (relative to 'path')  16551 images
    - images/train
  val: # val images (relative to 'path')  4952 images
    - images/val
  test: # test images (optional)
    - images/test

  # Classes
  # ['Chave', 'DISJUNTOR', 'TP', 'Pararraio', 'TC']
  names:
    0: person

  ```

  yolov8.yaml

  ```yaml
  # Ultralytics YOLO 🚀, AGPL-3.0 license
  # YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

  # Parameters
  nc: 80 # number of classes
  scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
    # [depth, width, max_channels]
    n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

  # YOLOv8.0n backbone
  backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]] # 9

  # YOLOv8.0n head
  head:
    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 6], 1, Concat, [1]] # cat backbone P4
    - [-1, 3, C2f, [512]] # 12

    - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
    - [[-1, 4], 1, Concat, [1]] # cat backbone P3
    - [-1, 3, C2f, [256]] # 15 (P3/8-small)

    - [-1, 1, Conv, [256, 3, 2]]
    - [[-1, 12], 1, Concat, [1]] # cat head P4
    - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

    - [-1, 1, Conv, [512, 3, 2]]
    - [[-1, 9], 1, Concat, [1]] # cat head P5
    - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

    - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)

  ```

  后面的 `data` 、 `engine` 、 `hub` 、 `models` 、 `nn` 、 `solution` 、 `trackers` 、 `utils` 则分别定义了数据、模型训练引擎、训练可视化、模型、网络结构、解决方案、追踪和工具类的底层代码实现。比如在nn目录下的models目录下的block.py中就给出了c2f模块的定义。

  ```python
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

- ultralytics.egg-info/

  **egg-info是一个目录** ，它在安装或开发Python包时由setuptools生成，用于存储关于该包的元数据。这些元数据对于包的管理、分发和安装至关重要，它们帮助pip和其他工具了解包的详细信息、版本信息、依赖关系等。这个目录是我们执行pip指令以开发者模式安装的时候出现的。

- CITATION.cff

  这个文件中包含了引用这个项目的格式说明。

- CONTRIBUTING.md

  这个里面说明了你可以如何为这个项目提供自己的贡献，让自己的名字出现在作者名单中。

- LICENSE

  提供了这该项目的许可信息。

- mkdocs.yml

  用于定义项目的各种设置和配置选项。

- pyproject.toml

`pyproject.toml` 是一个在 Python 项目中广泛使用的配置文件，它主要用于定义项目的构建系统要求、依赖关系以及相关的工具配置。现在这个 pyproject.toml 也是官方从PEP 518开始推荐的项目配置方式，感兴趣的小伙伴可以去看一下poetry这个库，通过poetry new可以很方便快捷的生成一个项目的脚手架代码。

- README.md

说明文件，也就是你现在看到的这个内容。

- README.zh-CN.md

中文格式的说明文件。

### 数据集准备

我们在这里准备了一系列标注好的数据集，如果大家不想自己标注可以看看这里是否有你需要得： [肆十二大作业系列清单-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/141068765) 
现在来到我们数据集准备的章节，这个章节教会大家如何自己构建一个yolo的检测数据以及如何使用。

首先这个位置提供了我们本次教程中使用到的一份标准的行人检测的数据集，一个标准的数据集的构成包含下面的几个目录。

```bash
├─images    # 图像文件夹
│  ├─train  # 训练集图像
│  ├─val    # 验证集图像
│  └─test   # 测试集图像
└─labels    # 标签文件夹，标签格式为yolo的txt格式
    ├─train # 训练集标签
    ├─val   # 验证集标签
    └─test  # 测试集标签
```

以训练集为例，给小伙伴们展示一下一个正确的对应关系是怎样的，这里训练集图像数量和标签数量是一致的，并且名称上面去除后缀之后是一一对应的。

 ![image-20240818134219994](https://i-blog.csdnimg.cn/blog_migrate/93fe61cf5d00a36346925125c46cf11b.png)

以2007_000480.jpg为例，下面是对应的2007_000480.txt中的标注文件内容。这个标注文件有三行，对应的是图像中的3个人物，其中每行包含5个数字，分别是类别，归一化处理之后目标中心点的x坐标、y坐标、目标归一化处理之后宽度w和目标归一化处理之后的宽度h。

 ![image-20240818134331149](https://i-blog.csdnimg.cn/blog_migrate/a72c28c7524f2f77441172c68bb09ed4.png)

如果大家要标注自己的数据集，则可以使用labelimg来进行标注，标注之前有几点注意事项。

- 标注的图像和路径尽量不要包含中文，图像名称最好是只有数字或者英文。

- 标注的时候尽量使用jpg的格式，如果是gif一类的格式后续可能有其他的麻烦。

- 标注的时候请一定要确保选择了yolo格式，如果不是yolo格式后续处理起来会非常麻烦。

- 最好一次性标注完，负责可能导致前后两次标签的结果不一致。

ok，首先大家可以在你的虚拟环境中通过 `pip install labelimg` 的指令安装标注然后，然后在命令行中键入labelimg来启动标注软件，如下图所示。

 ![image-20240818134903915](https://i-blog.csdnimg.cn/blog_migrate/93b08d365d57f6a62639b4d24e0a549b.png)

 ![image-20240818134948284](https://i-blog.csdnimg.cn/blog_migrate/10df05d13fc5a87ecd4c75b9b757e52e.png)

接下来，我们打开要标注的文件夹就可以进行标注了，标注之后一定要检查是否标注文件为txt格式。

 ![image-20240818141821358](https://i-blog.csdnimg.cn/blog_migrate/355d75c92d6974f1e611499aaf735bf8.png)

下面是labelimg的常用的快捷方式，大家可以熟悉下面的快捷方式帮助你提升标注的效率。

| ctrl+s | 保存 |
|:---:|:---:|
| ctrl+d | 复制当前标签和矩形框 |
| Ctrl + r | 更改默认注释目录（xml文件保存的地址） |
| Ctrl + u | 加载目录中的所有图像，鼠标点击Open dir同功能 |
| w | 创建矩阵 |
| d | 下一张 |
| a | 上一张 |
| delete | 删除选定的矩阵框 |
| space | 将当前图像标记为已标记 |

### 环境配置

来到我们最熟悉的章节，环境配置，老生常谈，环境配置基本是一致的，开始前请先学习pycharm和anaconda的使用，不熟悉的小伙伴可以移步这个位置： [【大作业系列入门教程】如何使用Anaconda和Pycharm_2024毕设系列如何使用anaconda-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/151867100) 

下载安装包到本地，首先请执行下列指令确保你已经配置好了国内的源。

```bash
conda config --remove-key channels
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

之后请先安装pytorch，请根据你设备的实际情况来选择执行GPU安装的指令或者是CPU安装的指令。

```bash
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
```

接着来到项目的目录下，以开发者的模式安装其余需要的库。

```
pip install -v -e .
```

 ![image-20240819003030009](https://i-blog.csdnimg.cn/blog_migrate/ed02ce78630be53acee0f7025f83d52f.png)

OK，我们来做一个简单的测试，观看是否生效，如果你看到了下面的结果则说明你的环境配置没有问题。

 ![image-20240819003145006](https://i-blog.csdnimg.cn/blog_migrate/45a63df58518338d3473095fd02be97d.png)

如果没有出现上面的检测结果，则说明你的配置出现了问题，可以在评论区中留言，让小伙伴们一同给你解答。

### 模型训练和测试

#### 模型的训练

训练的脚本对应的是：

```python
import time
from ultralytics import YOLO


# yolov8n模型训练：训练模型的数据为'A_my_data.yaml'，轮数为100，图片大小为640，设备为本地的GPU显卡，关闭多线程的加载，图像加载的批次大小为4，开启图片缓存
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
results = model.train(data='A_my_data.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=4, cache=True)  # 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用


```

开始之前请配置好你的数据集的路径和图片名称，比如我们今天训练的行人检测的数据集的配置如下：

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../../person_42_yolo_format
train: # train images (relative to 'path')  16551 images
  - images/train
val: # val images (relative to 'path')  4952 images
  - images/val
test: # test images (optional)
  - images/test

# Classes
# ['Chave', 'DISJUNTOR', 'TP', 'Pararraio', 'TC']
names:
  0: person

```

配置好了直接右键运行即可，比如我们这里是训练100轮，训练的过程中进度条会实时进行更新。

 ![image-20240818152211336](https://i-blog.csdnimg.cn/blog_migrate/2ddcaa2ec2aa44b87ff6747ca43aeea9.png)

训练完毕之后，将会在runs目录下生成一系列的结果图像。

 ![image-20240818152553473](https://i-blog.csdnimg.cn/blog_migrate/06dc05de030b22efac4cfb1b0f1ba4fe.png)

其中如果你需要在你的报告中说明模型的训练过程和训练结果，使用最多的分别是results.png和PR_curve.png，如下图所示。

 ![image-20240818152907734](https://i-blog.csdnimg.cn/blog_migrate/2376440b84d87c24da67e3caa63a0fd8.png)

 ![image-20240818152924610](https://i-blog.csdnimg.cn/blog_migrate/6fbd435ec1ea3880bdea1309d1508024.png)

#### 模型的测试

测试的脚本对应的是：

```python
from ultralytics import YOLO

# 加载自己训练好的模型，填写相对于这个脚本的相对路径或者填写绝对路径均可
model = YOLO("runs/detect/yolov8n/weights/best.pt")

# 开始进行验证，验证的数据集为'A_my_data.yaml'，图像大小为640，批次大小为4，置信度分数为0.25，交并比的阈值为0.6，设备为0，关闭多线程（windows下使用多线程加载数据容易出现问题）
validation_results = model.val(data='A_my_data.yaml', imgsz=640, batch=4, conf=0.25, iou=0.6, device="0", workers=0)

```

执行测试的脚本，将会输出下面的验证指标。

如果你不需要图形化界面，可以通过下面的脚本来直接对图像进行预测。

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["images/resources/demo.jpg", ])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="images/resources/result.jpg")  # save to disk

```

 ![image-20240818151945707](https://i-blog.csdnimg.cn/blog_migrate/99a0cce258279c9470cb0763157654be.png)

### 图形化界面开发

图形化界面部分的开发，我们采用了新的PySide6/PyQt6。PyQt和PySide都是C++的程序开发框架Qt的Python实现。PyQt是第三方组织对Qt官方提供的Python实现，也是Qt for Python最主要的实现。Qt官方对Python的支持力度越来越大，由于各种原因，Qt的官方选择使用PySide提供对Python Qt的支持。所以，Python Qt实际上存在两个分支：Qt4对应PyQt4和PySide；Qt5对应PyQt5和PySide2；Qt6则对应了PyQt6和PySide6。由于官方提供的PySide6从功能上来说更强，所以我们还是切换为PySide6作为本次图形化界面开发的框架。（实测下来他们直接的切换对于我们这些开发人员来说非常丝滑，我本次的更新中也只是更换了一下导入的过程）

下面是我们这个图形化界面程序的实现，值得注意的是，我们的图形化界面没有使用到designer的图形化工具，所以我们的代码是没有UI文件的，如果有小伙伴对designer那种可视化的图形化界面比较感兴趣，可以去学习一下那种方式，构建出来的图形化界面可能比较炫一些。

```python
import copy                      # 用于图像复制
import os                        # 用于系统路径查找
import shutil                    # 用于复制
from PySide6.QtGui import *      # GUI组件
from PySide6.QtCore import *     # 字体、边距等系统变量
from PySide6.QtWidgets import *  # 窗口等小组件
import threading                 # 多线程
import sys                       # 系统库
import cv2                       # opencv图像处理
import torch                     # 深度学习框架
import os.path as osp            # 路径查找
import time                      # 时间计算
from ultralytics import YOLO     # yolo核心算法

# 常用的字符串常量
WINDOW_TITLE ="Target detection system"
WELCOME_SENTENCE = "欢迎使用基于yolov8的行人检测系统"
ICON_IMAGE = "images/UI/lufei.png"
IMAGE_LEFT_INIT = "images/UI/up.jpeg"
IMAGE_RIGHT_INIT = "images/UI/right.jpeg"


class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)       # 系统界面标题
        self.resize(1200, 800)           # 系统初始化大小
        self.setWindowIcon(QIcon(ICON_IMAGE))   # 系统logo图像
        self.output_size = 480                  # 上传的图像和视频在系统界面上显示的大小
        self.img2predict = ""                   # 要进行预测的图像路径
        # self.device = 'cpu'
        self.init_vid_id = '0'  # 摄像头修改
        self.vid_source = int(self.init_vid_id)
        self.cap = cv2.VideoCapture(self.vid_source)
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model_path = "yolov8n.pt"  # todo 指明模型加载的位置的设备
        self.model = self.model_load(weights=self.model_path)
        self.conf_thres = 0.25   # 置信度的阈值
        self.iou_thres = 0.45    # NMS操作的时候 IOU过滤的阈值
        self.vid_gap = 30        # 摄像头视频帧保存间隔。
        self.initUI()            # 初始化图形化界面
        self.reset_vid()         # 重新设置视频参数，重新初始化是为了防止视频加载出错

    # 模型初始化
    @torch.no_grad()
    def model_load(self, weights=""):
        """
        模型加载
        """
        model_loaded = YOLO(weights)
        return model_loaded

    def initUI(self):
        """
        图形化界面初始化
        """
        # ********************* 图片识别界面 *****************************
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap(IMAGE_LEFT_INIT))
        self.right_img.setPixmap(QPixmap(IMAGE_RIGHT_INIT))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addWidget(self.right_img)
        self.img_num_label = QLabel("当前检测结果：待检测")
        self.img_num_label.setFont(font_main)
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
        img_detection_layout.addWidget(self.img_num_label)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # ********************* 视频识别界面 *****************************
        vid_detection_widget = QWidget()
        vid_detection_layout = QVBoxLayout()
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
        self.vid_num_label = QLabel("当前检测结果：{}".format("等待检测"))
        self.vid_num_label.setFont(font_main)
        vid_detection_layout.addWidget(self.vid_num_label)
        vid_detection_layout.addWidget(self.webcam_detection_btn)
        vid_detection_layout.addWidget(self.mp4_detection_btn)
        vid_detection_layout.addWidget(self.vid_stop_btn)
        vid_detection_widget.setLayout(vid_detection_layout)
        # ********************* 模型切换界面 *****************************
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel(WELCOME_SENTENCE)
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/zhu.jpg'))
        self.model_label = QLabel("当前模型：{}".format(self.model_path))
        self.model_label.setFont(font_main)
        change_model_button = QPushButton("切换模型")
        change_model_button.setFont(font_main)
        change_model_button.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(2,110,180);}"
                                          "QPushButton{background-color:rgb(48,124,208)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}"
                                          "QPushButton{margin:5px 5px}")

        record_button = QPushButton("查看历史记录")
        record_button.setFont(font_main)
        record_button.clicked.connect(self.check_record)
        record_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        change_model_button.clicked.connect(self.change_model)
        about_img.setAlignment(Qt.AlignCenter)
        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>作者：肆十二</a>")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addWidget(self.model_label)
        about_layout.addStretch()
        about_layout.addWidget(change_model_button)
        about_layout.addWidget(record_button)
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)
        self.left_img.setAlignment(Qt.AlignCenter)

        self.addTab(about_widget, '主页')
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(vid_detection_widget, '视频检测')

        self.setTabIcon(0, QIcon(ICON_IMAGE))
        self.setTabIcon(1, QIcon(ICON_IMAGE))
        self.setTabIcon(2, QIcon(ICON_IMAGE))
```

我们在代码的这个位置可以更换这个系统的默认标题和默认的logo图像。

 ![image-20240818144216340](https://i-blog.csdnimg.cn/blog_migrate/d20944634c4d4b396161a73a3a133747.png)

以及在这个位置可以更换为你自己的模型。

 ![image-20240818144234499](https://i-blog.csdnimg.cn/blog_migrate/e93bb183c872ca3a654344fb35450458.png)

