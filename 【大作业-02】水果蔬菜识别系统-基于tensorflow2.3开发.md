# 【大作业-02】水果蔬菜识别系统-基于tensorflow2.3开发

------------------------------------------------2021年6月18日重大更新--------------------------------------------------------------

目前已经退出bug修复之后的tensorflow2.3物体分类代码，大家可以训练自己的数据集，快来试试吧

csdn教程链接： [手把手教你用tensorflow2.3训练自己的分类数据集_CSDN博客](https://blog.csdn.net/ECHOSON/article/details/117964477) 

b站视频链接： [手把手教你用tensorflow2训练自己的数据集](https://www.bilibili.com/video/BV1rX4y1A7N8/) 

数据集链接： [计算机视觉数据集清单-附赠tensorflow模型训练和使用教程_CSDN博客](https://blog.csdn.net/ECHOSON/article/details/117964438?spm=1001.2014.3001.5502) 

代码链接： [vegetables_tf2.3: 基于tensorflow2.3开发的水果蔬菜识别系统 (gitee.com)](https://gitee.com/song-laogou/vegetables_tf2.3) 

------------------------------------------------------------------dejahu---------------------------------------------------------------------

tensorflow是由google出品的深度学习框架，2.x版本的tensorflow对代码进行了重构，使用起来更加简洁方便。本系统使用tensorflow训练了一个两层卷积两层池化的卷积神经网络和一个基于迁移学习的mobilnet网络，其中mobilenet网络模型准确率高达97%，可识别[‘土豆’, ‘圣女果’, ‘大白菜’, ‘大葱’, ‘梨’, ‘胡萝卜’, ‘芒果’, ‘苹果’, ‘西红柿’, ‘韭菜’, ‘香蕉’, ‘黄瓜’]12种水果蔬菜，并借助pyqt5构建了图形化界面，用户可选择图片上传进行果蔬的识别。

### 配套视频

配套视频在这里： [手把手教你用tensorflow2构建果蔬识别系统](https://www.bilibili.com/video/BV1q64y167xX) 

环境配置请看： [Python学习中Anaconda和Pycharm的正确打开方式_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV15o4y127Rt/) 

### 效果展示

首先来看看效果展示

启动系统，首先来到的是关于界面，作者的信息可以在window.py的84行进行修改。

 ![image-20210524104017908](https://i-blog.csdnimg.cn/blog_migrate/120d71a6ab2fada4d2015f4548173609.png)

主页部分，一共是两个按钮，点击 `上传图片` 的按钮可从本地选择图片进行上传

 ![image-20210524104702278](https://i-blog.csdnimg.cn/blog_migrate/5981ae1521049761837a181613ec61c0.png)

点击 `开始识别` 的按钮可以调用训练好的模型进行图片的识别，识别结果会显示在右侧

 ![image-20210524104625086](https://i-blog.csdnimg.cn/blog_migrate/d6763934667833013ff9701e34528278.png)

### 模型训练

模型的训练包含两部分代码， `train_mobilnet.py` 用来训练mobilnet模型， `train_cnn.py` 用于训练cnn模型，模型训练分为3个步骤：数据加载 - 模型加载 - 模型训练和保存。

##### 数据加载

首先来看一下我们的数据，我们分为训练集和测试集。

 ![image-20210524141444140](https://i-blog.csdnimg.cn/blog_migrate/40c3476a10facbea348ee9578d84ef0e.png)

打开之后按照目录存放相应的图片，比如大白菜目录下存放的全是白菜的图片。

 ![image-20210524141522571](https://i-blog.csdnimg.cn/blog_migrate/f82c4504ab8d82a801c7c8d6f08f054a.png)

这里我们使用 `tf.keras.preprocessing.image_dataset_from_directory` 直接从指定的目录中加载数据集并统一处理为指定的大小，代码如下：

```python
# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names
```

##### 模型构建

模型的构建方面我们借助keras来完成，只需要讲需要的层叠加到一起，并指明我们所需的优化器和损失函数即可

```python
# 构建CNN模型
def model_load(IMG_SHAPE=(224, 224, 3), class_num=6):
    # 搭建模型
    model = tf.keras.models.Sequential([
        # 对模型做归一化的处理，将0-255之间的数字统一处理到0到1之间
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        # 卷积层，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # 添加池化层，池化的kernel大小是2*2
        tf.keras.layers.MaxPooling2D(2, 2),
        # Add another convolution
        # 卷积层，输出为64个通道，卷积核大小为3*3，激活函数为relu
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 池化层，最大池化，对2*2的区域进行池化操作
        tf.keras.layers.MaxPooling2D(2, 2),
        # 将二维的输出转化为一维
        tf.keras.layers.Flatten(),
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),
        # 通过softmax函数将模型输出为类名长度的神经元上，激活函数采用softmax对应概率值
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model
```

##### 训练开始

训练的主流程中，先加载数据，再加载模型，最后将数据送入模型中进行训练，使用的是 `model.fit` 函数，可以自由指定训练的轮数，最终模型会保存在models目录下，训练过程中的可视化结果会保存在reuslts目录下

```python
def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # 加载数据集
    train_ds, val_ds, class_names = data_load("../data/vegetable_fruit/image_data",
                                              "../data/vegetable_fruit/test_image_data", 224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # 保存模型
    model.save("models/cnn_fv.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    # 绘制模型训练过程图
    show_loss_acc(history)
```

 ![image-20210524142945905](https://i-blog.csdnimg.cn/blog_migrate/5176a700bb8d969ea24368ca5d5c913b.png)

### 模型测试

模型测试和模型训练基本一致，只是在模型加载的部分，直接调用保存好的模型即可，不在进行模型参数的调整，以cnn模型的测试为例，首先加载数据，然后加载模型，最后使用 `model.evaluate` 方法对模型进行测试。

```python
def test_cnn():
    # 加载数据集
    train_ds, val_ds, class_names = data_load("../data/vegetable_fruit/image_data",
                                              "../data/vegetable_fruit/test_image_data", 224, 224, 16)
    # 加载模型
    model = tf.keras.models.load_model("models/cnn_fv.h5")
    # model.summary()
    # 测试
    loss, accuracy = model.evaluate(val_ds)
    # 输出结果
    print('CNN test accuracy :', accuracy)

```

测试结果如下，其中mobilenet可达到97%的准确率：

 ![image-20210524143235563](https://i-blog.csdnimg.cn/blog_migrate/c20d79751bc356ced192b24809287c39.png)

### 代码获取

代码请在码云中下载

[vegetables_tf2.3: 基于tensorflow2.3开发的水果蔬菜识别系统 (gitee.com)](https://gitee.com/song-laogou/vegetables_tf2.3) 

数据集请在csdn下载

[果蔬识别数据集.zip-专业指导文档类资源-CSDN下载](https://download.csdn.net/download/ECHOSON/19404659) 

下载之后按照下面的目录放置数据集和代码
 ![image-20210605170205728](https://i-blog.csdnimg.cn/blog_migrate/9f4544f5f859eef77f4621c0a37b4025.png)

`data` 目录下是

 ![image-20210605170318893](https://i-blog.csdnimg.cn/blog_migrate/6ab78bc38de727bec40d15e3ca3e0708.png)

`fv_tf2.3_cpu` 目录下是

 ![image-20210605170337870](https://i-blog.csdnimg.cn/blog_migrate/15c2b69bf42aeae960af3b3267e58eb7.png)

### 参考资料

李老师讲解卷积神经网络: https://www.bilibili.com/video/BV1Lb411b7BS?from=search&seid=14069382285256773171
如何理解卷积神经网络（CNN）中的卷积和池化:https://www.zhihu.com/question/49376084/answer/712089980
轻量级CNN网络之MobileNetv2：https://zhuanlan.zhihu.com/p/52426865
vgg网络解析：https://zhuanlan.zhihu.com/p/41423739
resnet网络解析：https://zhuanlan.zhihu.com/p/67860570

### 环境配置

环境的配置分为三步: 配置虚拟环境和安装程序所需要的包以及在pycharm中打开项目

#### 配置虚拟环境

配置虚拟环境需要通过anaconda来完成，anaconda的下载地址为：https://docs.conda.io/en/latest/miniconda.html

windows用户下载python3.8的miniconda即可

 ![image-20210524105054366](https://i-blog.csdnimg.cn/blog_migrate/c32e1ac821975f8f6acac9b15006f06f.png)

下载完毕之后双击安装即可，注意一点这些一定要选中

 ![image-20210524105240779](https://i-blog.csdnimg.cn/blog_migrate/1b8cbe88d1fc07fcd0748a1014888ab4.png)

 ![image-20210524105344251](https://i-blog.csdnimg.cn/blog_migrate/bf1cf0b58cdff6dab16d7c995ac3522d.png)

程序安装完毕之后打开windows的命令行（cmd），输入conda env list，出现下列信息则表示conda已完成安装

 ![image-20210524105549465](https://i-blog.csdnimg.cn/blog_migrate/498c05c6c4156520fc9e023426a6b69c.png)

在命令行中输入下列指令创建虚拟环境

```
conda create -n tf2.3 python==3.7.3
```

这条指令的含义是创建python版本为3.7.3，名称为tf2.3的虚拟环境

 ![image-20210524105851905](https://i-blog.csdnimg.cn/blog_migrate/5f56fd2c6e4bc89d549c6d7539820901.png)

安装结束之后输入下列指令激活虚拟环境，出现下图所示的小括号表示环境激活成功

```
conda activate tf2.3 
```

 ![image-20210524110020460](https://i-blog.csdnimg.cn/blog_migrate/eff1d6175b6c88e4ed126ea41967ea3d.png)

#### 安装程序所需要的包

接着在命令行中依次执行下列命令安装程序所需的包（请根据你的程序自由选择要安装的包）

```
pip install tensorflow-cpu==2.3.0 -i https://mirror.baidu.com/pypi/simple
pip install pyqt5 -i https://mirror.baidu.com/pypi/simple
pip install pillow -i https://mirror.baidu.com/pypi/simple 
pip install opencv-python -i https://mirror.baidu.com/pypi/simple 
pip install matplotlib -i https://mirror.baidu.com/pypi/simple 
```

 ![image-20210524110423033](https://i-blog.csdnimg.cn/blog_migrate/d3bd2f448e717533d73e124c0017cef4.png)

注：每一行命令都要执行

#### 在pycharm中打开程序

为了方便查看和调试代码，我们这里使用pycharm，pycharm的下载地址为：https://www.jetbrains.com/pycharm/download/#section=windows

 ![image-20210524110552220](https://i-blog.csdnimg.cn/blog_migrate/7749d83ad18050fe773004fd436f42e5.png)

这里选择社区版进行下载即可

安装过程中这几个选项请务必选上，以避免后面不必要的麻烦

 ![image-20210524111504457](https://i-blog.csdnimg.cn/blog_migrate/46a20e8852315b6d90765ffcd9c8d9ed.png)

安装完毕之后找到我们的代码，通过pycharm打开即可

 ![image-20210524111754027](https://i-blog.csdnimg.cn/blog_migrate/b49891601fb1abe92b70ed0492769621.png)

打开之后我们在软件的右下角找到 `Add interpreter` 添加刚才建立的虚拟环境

 ![image-20210524111938905](https://i-blog.csdnimg.cn/blog_migrate/2b958ae89d36e10d8b99dfd5461246d8.png)

 ![image-20210524112020699](https://i-blog.csdnimg.cn/blog_migrate/e4cebafcc7e8c83e1e94470d2c338bef.png)

最后右下角显示tf2.3表示我们当前的虚拟环境为tf2.3

 ![image-20210524112057673](https://i-blog.csdnimg.cn/blog_migrate/2fe6dffd5e02e79719573763b00b3d6b.png)

之后再安装程序的包的时候，直接在pycharm下方打开terminal，执行 `pip install xxx` 的指令即可，包就会自动安装在当前的虚拟环境中

 ![image-20210524112334043](https://i-blog.csdnimg.cn/blog_migrate/8345bda380a2a49d62a96c6e0b2a8668.png)

