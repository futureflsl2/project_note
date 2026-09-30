# 【肆十二】YOLO系列代码环境配置统一流程

## 【肆十二】YOLO系列代码环境配置统一流程

大家好，这里是肆十二，YOLO系列代码我们已经开发了很多内容，每次的内容除了要识别的内容不一致之外，其他环境配置一类的流程基本一致，为了提高效率，也为了节省大家的时间，我在本期将环境配置的统一问题进行抽取出来，后续的更新大家可以先看是否是你需要的内容，然后再来看环境配置的部分。

### 安装前的准备工作

安装之后需要你提前下载好Pycharm和miniconda，对这部分内容存在疑问的小伙伴可以看这期内容：

[Python项目配置前的准备工作_pyenv设置镜像-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/151867100?spm=1011.2415.3001.5331) 

### 环境配置

- 配置镜像

  miniconda默认的依赖下载位置在国内，采用默认的下载位置将会导致下载速度极慢。

  所以我们这里统一配置为清华的镜像，在你的命令行中执行下列的命令即可。

  ```bash
  conda config --remove-key channels
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  conda config --set show_channel_urls yes
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```

  执行的截图如下：

   ![image-20250131142121621](https://i-blog.csdnimg.cn/img_convert/1b6463b9543ecae33233bc9d0f179967.png)

- 新建虚拟环境

  虚拟环境是非常好用的，这样可以保证你的每个项目之间互相不会影响。所以我们这里建立一个虚拟环境，考虑到我们项目的时效性，这里我将会创建一个名字为 `yoloyolo` ，python版本为 `3.10.0` 的虚拟环境，创建虚拟环境之后激活虚拟环境。

  ```
  conda create -n yoloyolo python==3.10.0
  conda activate yoloyolo
  ```

  执行的截图如下。

   ![image-20250131142431760](https://i-blog.csdnimg.cn/img_convert/e16fd1620876260bc21570cca2c5fc6b.png)

   ![image-20250131142532991](https://i-blog.csdnimg.cn/img_convert/25e013f4acc9503e09930f927693dca7.png)

- 安装torch

  torch我们一般最好是单独安装，因为torch需要一些cuda之类的资源库，安装之前请保证你已经安装了比较新的nvidia驱动。注意只需要更新你的nvidia驱动即可，不需要安装其他的cuda或者cudnn之类的额外安装包。下面的命令分为两条，一条是适合有Nvidia显卡的小伙伴安装GPU版本的torch，一部分则是适合没有GPU安装cpu版本Torch的小伙伴。

  ```bash
  # CUDA 11.8
  conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
  # CUDA 12.1
  conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
  # CPU Only
  conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch
  ```

  下面是安装截图，我本地的电脑是nvidia的3060显卡。

   ![image-20250131142936708](https://i-blog.csdnimg.cn/img_convert/21da9d6ac838ea61d32a18c019dd2da7.png)

  我这里执行的是cuda12.1的版本的torch的安装。

   ![image-20250131143405733](https://i-blog.csdnimg.cn/img_convert/509a3a790765219ae47c891245574ba8.png)

  安装完了之后使用torch可能会有下列的警报信息。

   ![image-20250131143748696](https://i-blog.csdnimg.cn/img_convert/f9e0ea0d0e43bb74475c9dc990362ab6.png)

  所以我们需要在命令行中执行下列指令降低numpy的版本。

  ```bash
  pip install numpy==1.23.5
  ```

  安装之后可以进入python解释器执行下面的指令测试一下你的gpu安装是否可用

  ```python
  import torch
  flag = torch.cuda.is_available()
  print(f"GPU是否可用：{flag}")
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(device)
  print(torch.cuda.get_device_name(0))
  ```

  截图如下：

   ![image-20250131144032999](https://i-blog.csdnimg.cn/img_convert/7789e51ac92bb1c18a4fe305e07dcad0.png)

  到这里我们基本完成了torch的安装非常完美。

- 安装其他的依赖库

  接下来需要安装其他的依赖库的文件，所以在我们的教程中下载的代码文件中都包含一个 `project.toml` 的文件，这个文件中包含了我们需要安装的各种依赖库的版本，如下图所示。

   ![image-20250131144319896](https://i-blog.csdnimg.cn/img_convert/a1a4cd0706d89e4c529db4eca599654f.png)

  进行命令行之后，别忘记执行激活指令，激活我们已经安装了torch的虚拟环境。

  ```bash
  conda activate yolo
  ```

  然后执行pip指令来进行其他依赖库的安装。

  ```bash
  pip install -v -e .
  ```

   ![image-20250131144530590](https://i-blog.csdnimg.cn/img_convert/15add172e391caddd165c1fe1dcb9fdb.png)

   ![image-20250131145136609](https://i-blog.csdnimg.cn/img_convert/0ebb93e4449b6f7d66933c1600b0a07b.png)

- 进行简单的测试

  在我们的42_demo目录下面放了一段测试脚本，是使用yolo11模型进行视频文件的预测，如果你可以正常运行这段脚本，那表示你的安装是没有问题的。

  ```
  cd 42_demo
  python track_test.py
  ```

  如下图所示。

   ![image-20250131145442141](https://i-blog.csdnimg.cn/img_convert/805e9e583920396b1b6c9950ad04075d.png)

   ![image-20250131145357930](https://i-blog.csdnimg.cn/img_convert/23e881858fb41622f5e40651afe05910.png)

### 在Pycharm中运行项目

接下来我们使用pycharm打开项目，并配置我们刚才创建的虚拟环境。

 ![image-20250131145659141](https://i-blog.csdnimg.cn/img_convert/236a17363cc8b0b9dd98681cd6aae923.png)

选择虚拟环境

 ![image-20250131145722750](https://i-blog.csdnimg.cn/img_convert/c05e5a797cbffaddda517eafe461e1e1.png)

如果你的版本比较新，这样选择：

 ![image-20250131211358175](https://i-blog.csdnimg.cn/img_convert/c283235b59d225cf6e9e42f7afae5621.png)

如果你的版本比较老，这样选择。

 ![image-20250131145825853](https://i-blog.csdnimg.cn/img_convert/374e867b40a62f30f6d4a6aea841f81b.png)

等待加载完成，pycharm中成功出现虚拟环境。

 ![image-20250131145920855](https://i-blog.csdnimg.cn/img_convert/b6fdbae4447743099d416ff479083daf.png)

执行以下测试看是否可用。

 ![image-20250131145953191](https://i-blog.csdnimg.cn/img_convert/dc7fd45fad1918042ed7a28876b596c7.png)

出现下面的检测结果表示你可以直接起飞。

 ![image-20250131150033741](https://i-blog.csdnimg.cn/img_convert/992672eb0367312797e7905abedbd9fd.png)

### 其他

关于后续的训练和验证的过程你可以看每期的视频。

如果你想要通过视频的方式了解yolo的算法原理： [手把手教你使用YOLO11训练自己的检测追踪数据集_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1nzzdYwE2g/?vd_source=2f9a4e63109c3db3be5e8078e5111776) 

如果你需要了解yolo训练之后产生的各种指标图请看这个位置： [YOLO11模型指标解读-mAP、Precision、Recall_yolo11 怎么提高map-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/144097341) 

如果你还有其他的疑问可以看这个位置： [肆十二/contact_42](https://gitee.com/song-laogou/contact_42) 