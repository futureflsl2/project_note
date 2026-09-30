# 【大作业-01】花卉识别-基于tensorflow2.3实现

------------------------------------------------2024年12月17日更新-------------------------------------------------------------
目前已经距离这篇博客完成的时间过去了三年。在三年的时间里面，AI发生了巨大的变化，多模态、大模型的兴起，导致每个人在生活中都或多或少的接触到AI。
在各位小伙伴的支持下，目前我们的大作业专栏也已经更新了接近30期，因为每天来咨询的小伙伴比较多，我在这里把咨询的一些注意事项和常见的问题记录在这里，方便大家查看。

---------------》 [大作业系列常见问题答疑](https://gitee.com/song-laogou/contact_42) 
------------------------------------------------------------------肆十二---------------------------------------------------------------------
花卉识别是卷积神经网络的入门案例，这里我将模型的训练、测试、保存以及使用整合在了一起，如果看博客不是很理解这个项目在做什么，可以看下方的视频内容，我在视频中对原理和操作都进行了解析。



<div align="center" style="border: 3px solid gray;border-radius: 27px;overflow: hidden;"> <a class="link-info" href="https://player.bilibili.com/player.html?aid=373151373&autoplay=0" rel="nofollow" title="【大作业-01】基于tensorflow2.3的花卉识别程序">【大作业-01】基于tensorflow2.3的花卉识别程序</a><iframe id="oOg1lDVi-1734403980274" frameborder="0" src="https://player.bilibili.com/player.html?aid=373151373&amp;autoplay=0" allowfullscreen="true" data-mediaembed="bilibili" style="width: 100%; aspect-ratio: 2;" allow="fullscreen" loading="lazy"></iframe></div>



> [【大作业-01】基于tensorflow2.3的花卉识别程序_哔哩哔哩 (゜-゜)つロ 干杯~-bilibili](https://www.bilibili.com/video/BV1rZ4y1g7qh/) 

#### 数据集和模型下载地址

[【大作业-01】基于tensorflow和卷积神经网路的花卉识别系统.zip](https://download.csdn.net/download/ECHOSON/19687660?spm=1001.2101.3001.9500) 

### 文件目录

```bash
flower_tensorflow2.0
├─ data_read.py # 数据读取
├─ data_split.py # 数据切分
├─ images  # 图片文件
│    ├─ 123.jpg
│    ├─ init.png
│    ├─ logo.png
│    ├─ target.png
│    ├─ 主页面.png
│    └─ 关于.png
├─ window.py # ui界面
├─ models # 模型
│    ├─ cnn_flower.h5
│    └─ mobilenet_flower.h5
├─ readme.md 
├─ requirements.txt # 安装需求
├─ test_model.py # 模型测试
└─ train_model.py # 模型训练
```

### 如何使用

首先你需要下载项目到你的本地。

实践方面，学习深度学习离不开python，工欲善其事必先利其器，你需要掌握两个最基础的软件的使用，分别是用了管理python环境的anaconda和用于编辑调试代码的pycharm。

您可以通过这期视频对这两个软件进行学习： [【2024毕设系列】Anaconda和Pycharm如何使用_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1A2421F7VE/) 

初次之外，如果您在后续的学习和工作中经常使用到深度学习，那对于服务器的基本操作也需要掌握： [手把手教你使用服务器训练AI模型_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1TuxLeVED6/) 

确定你的电脑已经安装好了PyQt5、tensorflow2.0以及opencv-python等相关软件之后，你可以执行下列命令进行安装

```
cd flower_tensorflow2.3
conda create -n flower_demo 
pip install -r requirements.txt
```

如果你想要重新训练你的模型，请执行

```
python train_mobilenet.py
python train_cnn.py
```

训练的详细脚本如下：

```python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 20:29
# @Author  : 肆十二
# @Email   : 3045834499@qq.com
# @File    : train_mobilenet.py
# @Software: PyCharm
# @Brief   : mobilenet模型训练代码，训练的模型会保存在models目录下，折线图会保存在results目录下

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


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


# 构建mobilenet模型
# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), class_num=12):
    # 微调的过程中不需要进行归一化的处理
    # 加载预训练的mobilenet模型
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # 将模型的主干参数进行冻结
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        # 进行归一化的处理
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),
        # 设置主干模型
        base_model,
        # 对主干模型的输出进行全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),
        # 通过全连接层映射到最后的分类数目上
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    # 模型训练的优化器为adam优化器，模型的损失函数为交叉熵损失函数
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_mobilenet.png', dpi=100)


def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # todo 加载数据集， 修改为你的数据集的路径
    train_ds, val_ds, class_names = data_load("../split_data/train",
                                              "../split_data/val", 224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo 保存模型， 修改为你要保存的模型的名称
    model.save("models/mobilenet_new.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)

```

这里我提供了两个模型的训练，分别是基础的lenet模型和进阶的轻量化模型mobilenet，训练结束之后将会输出训练过程中的准确率和误差曲线，如下图所示。
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a1fdb1d996d54faba6e9d5ec5d674a5e.png)

如果你想要测试模型的准确率，请执行

```
python test_model.py
```

测试之后，将会输出模型准确率的混淆矩阵，如下图所示。
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97f7e63c3bba4e4bad6368c3abe24e3c.png)

如果你想看看图形化的界面，请执行

```
python window.py
```

### 执行效果

图形化界面
 ![image-20201212161743464](https://i-blog.csdnimg.cn/blog_migrate/7b30392e66a84f945d77f0c8fd015c28.png)

