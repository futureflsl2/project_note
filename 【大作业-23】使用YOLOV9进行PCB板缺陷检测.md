# 【大作业-23】使用YOLOV9进行PCB板缺陷检测

## 使用YOLOV9进行PCB板缺陷检测

<div align="center" style="border: 3px solid gray;border-radius: 27px;overflow: hidden;"> <a class="link-info" href="https://player.bilibili.com/player.html?aid=113244530283400&autoplay=0" rel="nofollow" title="【大作业-23】使用YOLOV9进行PCB板缺陷检测">【大作业-23】使用YOLOV9进行PCB板缺陷检测</a><iframe id="WA5B4D9i-1759238659573" frameborder="0" src="https://player.bilibili.com/player.html?aid=113244530283400&amp;autoplay=0" allowfullscreen="true" data-mediaembed="bilibili" style="width: 100%; aspect-ratio: 2;" allow="fullscreen" loading="lazy"></iframe></div>



本次教程依然是YOLOV8教程的延续，我们本次主要使用YOLOV9对PCB板进行缺陷检测，包含数据的处理、模型的训练、测试以及图形化系统的构建，在图形化的系统中我们通过点击按钮的形式即可完成对训练好的模型的调用。

 ![image-20241003213644908](https://i-blog.csdnimg.cn/img_convert/18d71b97bfd47556d3b7245ebe35292f.png)

### 如何执行该项目

为了防止有的小伙伴找不到项目配置的地方，我直接将项目配置放在了前面，后面是一些报告和实验结果的分析。

#### 环境配置

环境配置这块我这边只列举了指令，考虑到大部分好兄弟看的时候多少有点疑惑，这里我建议大家按照视频来进行执行更加方便。

- 镜像加速

  ```bash
  conda config --remove-key channels
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  conda config --set show_channel_urls yes
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```

- 创建和激活虚拟环境

  ```bash
  conda create -n torch python==3.8.5
  y
  conda activate torch
  ```

- 安装torch

  ```bash
  conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
  conda install pytorch==1.10.0 torchvision torchaudio cudatoolkit=11.3 # 30系列以上显卡gpu版本pytorch安装指令
  conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
  ```

- 在项目目录下执行其他库的安装

  ```
  # 注意这条指令一定要在你的项目目录下执行。
  pip install -v - e .
  ```

ok，使用pycharm打开之后并启动即可起飞。

#### 模型训练

模型训练之前请务必将您的数据集路径切换为您本地的数据集路径。

 ![image-20241003215131486](https://i-blog.csdnimg.cn/img_convert/f05172dea39e86af3a65c477bfe6e79f.png)

训练的代码如下，直接右键执行即可。

```bash
import time
from ultralytics import YOLO
# yolov8n模型训练：训练模型的数据为'A_my_data.yaml'，轮数为100，图片大小为640，设备为本地的GPU显卡，关闭多线程的加载，图像加载的批次大小为4，开启图片缓存
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")   # load a pretrained model (recommended for training)
model = YOLO("yolov9s.yaml").load("../../pretrained/yolov9s.pt")   # load a pretrained model (recommended for training)
results = model.train(data='A_my_data.yaml', epochs=100, imgsz=640, device=[0,], workers=0, batch=4, cache=True)  # 开始训练
time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用
```

 ![image-20241003214951039](https://i-blog.csdnimg.cn/img_convert/26a4378687be19d539a1d50a633b41ab.png)

#### 模型测试

测试的代码如下，换成你自己训练好的模型之后也是直接右键执行即可。

```
from ultralytics import YOLO

# 加载自己训练好的模型，填写相对于这个脚本的相对路径或者填写绝对路径均可
model = YOLO("runs/detect/yolov8n/weights/best.pt")

# 开始进行验证，验证的数据集为'A_my_data.yaml'，图像大小为640，批次大小为4，置信度分数为0.25，交并比的阈值为0.6，设备为0，关闭多线程（windows下使用多线程加载数据容易出现问题）
validation_results = model.val(data='A_my_data.yaml', imgsz=640, batch=4, conf=0.25, iou=0.6, device="0", workers=0)

```

 ![image-20241003215025567](https://i-blog.csdnimg.cn/img_convert/c1ce002483a8145b6a4722b69a5beb52.png)

#### 图形化界面开发

图形化界面我们依然使用的是pyside6进行开发，只需要在这个位置换成你训练好的模型路径即可。

 ![image-20241003214332795](https://i-blog.csdnimg.cn/img_convert/cb91c1a47f57f4f76793caad0abf994c.png)

右键运行启动之后即可上传图像或者视频进行检测。

 ![](https://i-blog.csdnimg.cn/img_convert/18d71b97bfd47556d3b7245ebe35292f.png)

### 背景和意义

电路板（Printed Circuit Board, PCB）是现代电子设备的核心组成部分，广泛应用于各种工业和消费类电子产品中。PCB制造的质量直接影响电子设备的性能和稳定性。然而，由于制造工艺复杂，电路板在生产过程中可能会产生各种缺陷，如焊点不良、短路、断路、元件错位等。这些缺陷如果不及时发现和修复，可能导致产品失效，甚至造成严重的经济损失。

传统的电路板缺陷检测依赖于人工检测或基于规则的机器视觉系统。人工检测耗时费力，效率低下且容易受人为因素影响。基于规则的视觉系统虽然能提高效率，但面对复杂和多变的缺陷类型，常常表现出不足。因此，近年来，深度学习技术，特别是基于卷积神经网络（CNN）的目标检测算法，逐渐成为解决这一问题的重要工具。

将YOLO应用于电路板缺陷检测具有以下几方面的意义：

1. **高效性和实时性** ：YOLO的端到端架构能够在一次前向传播中同时完成目标的定位和分类，这使得它在目标检测任务中具有极快的推理速度。对于工业生产中的在线缺陷检测，实时性是至关重要的。相比于其他方法，YOLO能够在保证精度的同时，以更快的速度进行缺陷检测，确保生产线的连续性。

2. **提高检测精度** ：YOLO经过多年的改进（如YOLOv5, YOLOv8等），已经能够在复杂背景下精确检测微小目标。在PCB检测任务中，许多缺陷较小且形态多样，YOLO能通过其多尺度检测能力，显著提高检测的精度和稳定性。

3. **智能化与自动化** ：基于YOLO的电路板缺陷检测可以大幅减少对人工的依赖，降低检测成本，并提高检测的客观性和一致性。同时，自动化检测系统能够与生产线无缝集成，显著提高生产效率。

4. **应对复杂缺陷类型** ：YOLO模型具备强大的泛化能力，能够检测多种类型的缺陷，无需为每一种缺陷类型编写单独的检测规则。这使得它在面对电路板上多种多样的缺陷时，表现出更好的适应性。

5. **推动智能制造发展** ：基于YOLO的电路板缺陷检测技术是智能制造领域的重要组成部分。它推动了工业生产向自动化、智能化方向发展，有助于提升电子制造业的整体技术水平，增强产业竞争力。

### 相关工作

电路板缺陷检测近年来得到了越来越多的关注，尤其是深度学习的兴起为这一领域带来了显著的进展。以下是近年来使用深度学习进行电路板缺陷检测的相关文献，并基于这些文献给出一个简要综述。

1. **Gao et al. (2019). “Automatic PCB Defect Detection Using Deep Learning and Unsupervised Feature Learning”** 
   该研究提出了一种基于深度学习的PCB缺陷检测方法，利用卷积神经网络（CNN）自动提取特征，并结合无监督学习方式对电路板缺陷进行检测。该方法不依赖于传统的人工特征工程，具有较强的鲁棒性和较高的检测精度。

2. **Zhao et al. (2020). “Deep Learning for PCB Defect Detection: A Review”** 
   这篇综述总结了近年来基于深度学习的PCB缺陷检测方法，包括不同的网络架构和数据处理方法，探讨了如何提高检测效率和精度。文章中强调了YOLO、Faster R-CNN等检测模型在这一任务中的成功应用。

3. **Li et al. (2021). “A Deep Learning Approach for Detecting PCB Defects Using Convolutional Neural Networks”** 
   本文提出了一种基于CNN的电路板缺陷检测方法，并通过实验展示了其在检测效率和精度方面的优势。研究还探讨了如何处理不平衡数据和小目标检测的问题。

4. **Kim et al. (2021). “Real-Time PCB Defect Detection Using YOLOv5”** 
   该文展示了YOLOv5在电路板缺陷检测中的应用，强调了其在实时检测方面的优越性。研究通过大量实验比较了不同YOLO版本的性能，并优化了模型参数以适应PCB的具体检测需求。

5. **Wang et al. (2023). “Transformer-Based PCB Defect Detection with Attention Mechanism”** 
   本文结合了Transformers和注意力机制，提出了一种新型的电路板缺陷检测模型。研究表明，与传统的CNN方法相比，基于Transformer的模型在检测复杂缺陷时具有更好的性能，特别是在处理形态复杂的缺陷和背景干扰较大的情况下。

过去五年来，深度学习技术，特别是卷积神经网络（CNN）和目标检测算法，在电路板缺陷检测中取得了显著进展。总体上，这些研究主要集中在以下几个方面：

1. **网络架构的发展** 

   - 经典的深度学习算法如Faster R-CNN和YOLO逐渐成为电路板缺陷检测中的主流工具。YOLO因其单次前向传播的高效性，在实时检测方面表现优异，特别是在大规模生产中有着广泛应用。而Faster R-CNN等两阶段算法则在精度上占据一定优势，但相对速度较慢，更多应用于离线检测。

   - 随着Transformer架构在视觉任务中的兴起，近年来一些研究探索了基于Transformer的模型，如Wang et al. (2023) 提出的注意力机制，大大提高了复杂背景下的缺陷检测精度。这类模型通过捕捉长距离依赖，展现出在检测形态复杂的缺陷方面的潜力。

2. **小目标检测与数据不平衡问题** 

   电路板缺陷往往属于小目标检测的范畴，并且不同种类的缺陷可能在数据中分布不均。为了解决这一问题，许多研究尝试通过数据增强、特征金字塔（如FPN）以及损失函数的调整来提升小目标检测的效果。此外，不平衡数据问题在大多数文献中被广泛讨论，诸如采样策略的优化和损失函数加权等技术手段被用于缓解这一问题。

3. **实时性与工业应用** 

   实时性一直是工业缺陷检测中的关键要求。YOLO家族模型（尤其是YOLOv5、YOLOv7等）在工业场景中表现出了极高的效率，能够在检测速度和检测精度之间取得良好平衡。Kim et al. (2021) 的研究表明，通过优化YOLOv5的超参数和模型结构，可以实现低延迟的缺陷检测，适用于流水线中的在线检测。

4. **无监督与半监督学习** 

   在实际生产中，缺陷数据往往难以标注，特别是某些罕见或未曾见过的缺陷类型。近年来，越来越多的研究开始关注无监督学习和半监督学习在电路板缺陷检测中的应用。Gao et al. (2019) 提出的无监督特征学习方法，通过自动提取PCB图像的潜在特征，在标注有限的情况下，依然能够获得较高的检测效果。

5. **数据集与迁移学习** 

   深度学习模型的性能往往依赖于大规模标注数据集。然而，电路板缺陷检测的领域数据集较为有限。为此，部分研究者尝试使用迁移学习方法，借助在通用数据集（如COCO、PASCAL VOC）上预训练的模型进行迁移，并在少量电路板数据上进行微调，取得了良好的效果。此外，生成对抗网络（GAN）等技术也被用于合成更多的缺陷样本，以扩大训练数据规模。

### 本文所使用方法-yolov9

在追求最优实时目标检测的过程中，YOLOv9凭借其创新的方式，成功克服了深度神经网络中固有的信息丢失挑战。通过集成可编程梯度信息（PGI）和通用高效层聚合网络（GELAN）架构，YOLOv9不仅增强了模型的学习能力，还确保了在检测过程中关键信息的保留，从而实现了卓越的准确性和性能。

结构图来源： [divided7/yolov9_structure_graph (github.com)](https://github.com/divided7/yolov9_structure_graph/tree/main) 

 ![image-20241003211838301](https://i-blog.csdnimg.cn/img_convert/88dbd3a5dbab61502241ffa9959ad070.png)

#### YOLOv9的核心创新

YOLOv9的进步深深根植于解决深度神经网络中信息丢失的挑战。信息瓶颈原理和可逆函数的创新使用是其设计的核心，确保了YOLOv9在保持高效率的同时也能保持出色的准确性。

#### 信息瓶颈原理

信息瓶颈原理揭示了深度学习中的一个基本挑战：随着数据通过网络的连续层，信息丢失的可能性增加。这一现象可以用数学公式表示为：

```python
I(X, X) >= I(X, f_theta(X)) >= I(X, g_phi(f_theta(X)))
```

其中， `I` 表示互信息， `f` 和 `g` 分别为带有参数 `theta` 和 `phi` 的变换函数。YOLOv9通过实现可编程梯度信息（PGI），应对这一挑战，帮助在网络的深度中保留关键信息，确保生成更可靠的梯度，从而提高模型的收敛性和性能。

#### 可逆函数

可逆函数的概念是YOLOv9设计的另一个基石。一个函数如果可以在没有任何信息丢失的情况下被反转，则被认为是可逆的，其表达式为：

```python
X = v_zeta(r_psi(X))
```

其中， `psi` 和 `zeta` 是可逆函数及其逆函数的参数。这一性质对于深度学习架构至关重要，因为它允许网络保留完整的信息流，从而使模型参数的更新更加准确。YOLOv9在其架构中引入了可逆函数，以降低深层次信息退化的风险，确保在目标检测任务中保留关键信息。

#### 对轻量模型的影响

对于轻量模型，解决信息丢失问题尤为重要，因为这些模型通常参数较少，容易在前向传播过程中丢失大量信息。YOLOv9的架构通过使用PGI和可逆函数，即使在精简的模型中，也能确保关键信息得到保留并被有效利用，从而实现准确的目标检测。

#### 可编程梯度信息（PGI）

PGI是YOLOv9引入的新概念，用于应对信息瓶颈问题，确保在深层网络中保留关键信息。这使得网络能够生成可靠的梯度，促进模型的准确更新，并提升整体检测性能。

 ![image-20240920175237018](https://i-blog.csdnimg.cn/img_convert/baea0a23872faf3c13c70707c3e900e3.png)

#### 通用高效层聚合网络（GELAN）

 ![image-20240920175300248](https://i-blog.csdnimg.cn/img_convert/9cbc8cd1ccc3b7ed7b403b630c6b831e.png)

 ![image-20241003215224035](https://i-blog.csdnimg.cn/img_convert/dfe894ab8047b0bdec63cc7b9cda789b.png)

GELAN代表了YOLOv9在架构上的战略性进步，它实现了更优的参数利用率和计算效率。其设计允许灵活集成各种计算模块，使YOLOv9能够适应广泛的应用场景，而不会牺牲速度或准确性。

### 实验和实验结果分析

实验结果全部保存在了这个目录下面，对于每个图的解释可以参考这里的视频： [手把手教你用YOLOv8训练自己的数据集（原理解析+代码实践）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1KHp2eREFZ/?vd_source=2f9a4e63109c3db3be5e8078e5111776) 

 ![image-20241003214435914](https://i-blog.csdnimg.cn/img_convert/cf5ead05124247001ac50fb5b0a6ea9f.png)

在你的报告中只需要将这些图依次贴出来然后进行说明即可。

我给你打个样，这里大家直接kimi就完事了，免费还好用。

从你提供的图片内容来看，这是一个关于YOLOv9模型在电路板缺陷检测任务中的精确度-召回率（Precision-Recall Curve）图和一些关键指标的汇总。

 ![image-20241003214748128](https://i-blog.csdnimg.cn/img_convert/4c49c8bb2b318a78b3f763e30723a5f7.png)

分析Precision-Recall Curve

1. **Precision（精确度）** ：预测为正的样本中实际为正的比例。

2. **Recall（召回率）** ：实际为正的样本中被预测为正的比例。

在PR曲线中：

- **x轴** 表示召回率。

- **y轴** 表示精确度。

从图中可以看出：

- 当召回率从0增加到1时，精确度从1逐渐下降到0。

- 曲线的形状通常反映了模型在不同阈值下的表现。

具体缺陷检测结果

表格中列出了不同缺陷类型及其对应的精确度：

- **missing_hole** ：0.995

- **mouse_bite** ：0.889

- **open_circuit** ：0.957

- **short** ：0.962

- **spurious_copper** ：0.925

- **spur** ：0.782

综合分析

1. **高精确度** ： `missing_hole` 和 `short` 的精确度非常高，接近1，说明模型在检测这些缺陷时非常准确。

2. **中等精确度** ： `open_circuit` 、 `spurious_copper` 和 `mouse_bite` 的精确度在0.89到0.96之间，表现良好。

3. **低精确度** ： `spur` 的精确度为0.782，相对较低，可能需要进一步优化。

平均精度（mAP@0.5）

- **mAP@0.5** ：所有类别的平均精度在0.5的IoU阈值下为0.918，这是一个相对较高的表现。

结论

YOLOv9在电路板缺陷检测任务中表现良好，尤其是在检测 `missing_hole` 和 `short` 缺陷时。对于 `spur` 缺陷，模型的精确度较低，可能需要进一步的数据增强或模型调优。

如果需要进一步优化模型，可以考虑以下方法：

1. **数据增强** ：增加更多 `spur` 缺陷的训练样本。

2. **模型调参** ：调整模型的超参数，如学习率、批大小等。

3. **后处理** ：使用更复杂的后处理方法来提高精确度。

希望这些分析对你有所帮助！如果还有其他问题，请随时告诉我。

### 结论和展望

#### 结论

将YOLOv9应用于电路板缺陷检测任务中，可以有效提升检测精度、速度以及对复杂缺陷类型的识别能力。YOLOv9的创新点，如可编程梯度信息（PGI）和通用高效层聚合网络（GELAN），成功解决了深度神经网络中的信息丢失问题，使得即使在小目标或背景复杂的电路板图像中，模型仍能保持较高的检测准确性。此外，YOLOv9的轻量化设计和可逆函数的应用，确保了在保持高检测效率的同时不牺牲信息的完整性。

在实际的电路板检测场景中，YOLOv9表现出了以下几方面的优势：

1. **高精度与低延迟** ：YOLOv9不仅在缺陷检测的准确性上表现出色，还能够以较低的推理时间实现实时检测，非常适合工业流水线的在线缺陷检测。

2. **强大的泛化能力** ：通过PGI和GELAN等架构创新，YOLOv9能够有效应对电路板上多种复杂缺陷的检测任务，涵盖了从焊点不良到微小的短路、断路等多种缺陷。

3. **模型轻量化** ：YOLOv9在不增加过多计算成本的前提下，保留了关键特征，确保了在嵌入式设备和资源有限的硬件环境中的可应用性。

#### 展望

未来，YOLOv9在电路板缺陷检测领域的应用有着广阔的前景，特别是在以下几个方面：

1. **更精细的多任务检测** ：YOLOv9未来可以扩展到不仅检测缺陷的位置和类型，还可以实现更精细的多任务目标检测，如同时检测多个缺陷或识别缺陷的严重程度。这将为更加全面的电路板质量检测提供技术支持。

2. **结合其他深度学习技术** ：YOLOv9的架构可以与其他深度学习技术（如Transformer、注意力机制）进一步结合，以提升其对复杂背景和微小目标的检测能力。特别是在应对高度复杂的电路板设计时，增强模型的上下文理解能力将大有裨益。

3. **无监督学习与少样本学习** ：未来，YOLOv9可以结合无监督学习或少样本学习方法，减少对大量标注数据的依赖。这对于处理少见或新出现的电路板缺陷特别有用，能够帮助模型自动学习新的缺陷特征，提升检测的广泛性和灵活性。

4. **工业4.0与智能制造的融合** ：YOLOv9在电路板缺陷检测中的应用有助于推进智能制造的发展。通过与工业物联网（IIoT）和自动化生产线的深度集成，YOLOv9可以为大规模生产提供实时、自动化的缺陷检测解决方案，显著提高生产效率和产品质量。

总的来说，YOLOv9不仅在当前的电路板缺陷检测中展示了强大的技术优势，也为未来的智能制造和自动化质量检测带来了更多的可能性。未来的研究可以在提升模型精度、减少数据依赖以及拓展实际应用场景等方面进行探索。