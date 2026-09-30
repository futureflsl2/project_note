# 【大作业-04】手把手教你构建垃圾分类系统-基于tensorflow2.3

> B站视频： [手把手教你构建垃圾分类系统-基于tensorflow2.3_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1kX4y1A7VT) 
> 
> 代码地址： [trash_classification_tf2.3: 基于tensorflow2.3的垃圾分类系统 (gitee.com)](https://gitee.com/song-laogou/trash_classification_tf2.3) 
> 
> 模型和数据集地址： [垃圾分类数据集和tf代码-8w张图片245个类.zip-深度学习文档类资源-CSDN下载](https://download.csdn.net/download/ECHOSON/19713816) 
> 
> 大家有什么想法也可以在评论区留言，我会根据留言情况出对应的教程。

哈喽，大家好，这里是dejahu，本期教程给大家带来的是 `基于tensorflow2.3的垃圾分类系统` ，主要还是为了教会大家如何使用我写好的物体分类模板代码去训练自己的数据集（ [手把手教你用tensorflow2.3训练自己的分类数据集_dejahu的博客-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/117964477) ）。本期构建的模型将会采用我事先清洗好的数据集，包含4个大类和245个小类，为了方便对比，我将会训练两个卷积神经网络模型，一个是结构比较简单的lenet模型，一个是结构相对复杂但是网络更加轻量的mobilnet模型，最后通过pyqt5构建图形化的界面，用户上传图片，系统给出垃圾的具体种类，效果如下：

 ![image-20210625133930243](https://i-blog.csdnimg.cn/blog_migrate/bb587947d4b22debb52fcf3e7c896382.png)

 ![image-20210625134014304](https://i-blog.csdnimg.cn/blog_migrate/ef4e9ad3442ed90ea6666110fb89d9b7.png)

这段话大家可以写在实验报告的前面，说明垃圾分类的意义。2019年9月，为推动全国公共机构做好生活垃圾分类工作，发挥率先示范作用，国家机关事务管理局印发通知，公布《公共机构生活垃圾分类工作评价参考标准》，并就进一步推进有关工作提出要求。垃圾分类在我国很多城市已经推行了很长的一段时间并且将会持续推行下去，垃圾分类可以提高垃圾的资源价值和经济价值，力争物尽其用，减少垃圾处理量和处理设备的使用，降低处理成本，减少土地资源的消耗，具有社会、经济、生态等几方面的效益。

### 环境配置

#### Ananconda和pycharm软件安装

我们的代码是基于python进行开发的，所以在正式开始之前，你需要保证你的电脑上已经配置好了python的基础环境。这里我们将会使用到两个软件：一个是anaconda，用来帮助我们控制虚拟环境，一个是pycharm，用来帮助我们编写和调试代码，关于pycharm和ananconda的具体使用我在这里就不再进行赘述了，不会使用这两个软件的小伙伴可以看这期教程，里面详细演示了如何正确使用ananconda和pycharm，以及通过更换源的方式来提高anaconda中下载第三方库的速度的方法。

> 环境配置博客地址： [如何在pycharm中配置anaconda的虚拟环境_dejahu的博客-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/117220445) 

在命令行中输出conda，如果输出下列信息，则表示ananconda安装完毕。

 ![image-20210625134126925](https://i-blog.csdnimg.cn/blog_migrate/c1175d4c7ff843635129e4f7918a525a.png)

另外pycharm我推荐大家使用最新的社区版本，免费而且功能强大，这是我本机的版本。

 ![image-20210625134140526](https://i-blog.csdnimg.cn/blog_migrate/a407e00dab9fa3076c580e4e5745c021.png)

#### 代码配套虚拟环境配置

首先从下方链接下载代码并进行解压：

> 代码地址： [trash_classification_tf2.3: 基于tensorflow2.3的垃圾分类系统 (gitee.com)](https://gitee.com/song-laogou/trash_classification_tf2.3) 

解压之后在代码目录下打开cmd

创建虚拟环境

 ![image-20210625153604709](https://i-blog.csdnimg.cn/blog_migrate/b742a031c6fa1e38fc3f3ce75556b7dc.png)

激活虚拟环境

 ![image-20210625153635327](https://i-blog.csdnimg.cn/blog_migrate/5d35d06717a41f701cddfd4ddd0fc7f8.png)

安装程序所需的第三方库

 ![image-20210625153656623](https://i-blog.csdnimg.cn/blog_migrate/5fbdb88c71316607211dc025d19d307a.png)

### 数据处理

> 模型和数据集地址： [垃圾分类数据集和tf代码-8w张图片245个类.zip-深度学习文档类资源-CSDN下载](https://download.csdn.net/download/ECHOSON/19713816) 

我通过爬虫的形式爬取了数据集并进行了清洗和分类，大家在下方的链接下载压缩包解压之后，可以在这个txt文件中获取该数据集的百度网盘链接，通过网盘下载之后进行解压即可：

 ![image-20210625134928624](https://i-blog.csdnimg.cn/blog_migrate/40f11567b64a0494b654e7618d6213b7.png)

解压之后的数据集如下图所示，每个子文件夹存放的分别是对应名称的垃圾图片。

 ![0ef0f54df64291822809b6ba7419af60](https://i-blog.csdnimg.cn/blog_migrate/794aa652e67c8c3c04006c369f274901.png)

这里数据集一共包含 `4个大类` 和 `245个小类` ，大概有8w张图片左右，四个大类分别是 `厨余垃圾` 、 `可回收物` 、 `有害垃圾` 和 `其他垃圾` ，245个小类的具体类别如下：

```
trash_jpg
├─ 其他垃圾_PE塑料袋
├─ 其他垃圾_U型回形针
├─ 其他垃圾_一次性杯子
├─ 其他垃圾_一次性棉签
├─ 其他垃圾_串串竹签
├─ 其他垃圾_便利贴
├─ 其他垃圾_创可贴
├─ 其他垃圾_卫生纸
├─ 其他垃圾_厨房手套
├─ 其他垃圾_厨房抹布
├─ 其他垃圾_口罩
├─ 其他垃圾_唱片
├─ 其他垃圾_图钉
├─ 其他垃圾_大龙虾头
├─ 其他垃圾_奶茶杯
├─ 其他垃圾_干燥剂
├─ 其他垃圾_彩票
├─ 其他垃圾_打泡网
├─ 其他垃圾_打火机
├─ 其他垃圾_搓澡巾
├─ 其他垃圾_果壳
├─ 其他垃圾_毛巾
├─ 其他垃圾_涂改带
├─ 其他垃圾_湿纸巾
├─ 其他垃圾_烟蒂
├─ 其他垃圾_牙刷
├─ 其他垃圾_电影票
├─ 其他垃圾_电蚊香
├─ 其他垃圾_百洁布
├─ 其他垃圾_眼镜
├─ 其他垃圾_眼镜布
├─ 其他垃圾_空调滤芯
├─ 其他垃圾_笔
├─ 其他垃圾_胶带
├─ 其他垃圾_胶水废包装
├─ 其他垃圾_苍蝇拍
├─ 其他垃圾_茶壶碎片
├─ 其他垃圾_草帽
├─ 其他垃圾_菜板
├─ 其他垃圾_车票
├─ 其他垃圾_酒精棉
├─ 其他垃圾_防霉防蛀片
├─ 其他垃圾_除湿袋
├─ 其他垃圾_餐巾纸
├─ 其他垃圾_餐盒
├─ 其他垃圾_验孕棒
├─ 其他垃圾_鸡毛掸
├─ 厨余垃圾_八宝粥
├─ 厨余垃圾_冰激凌
├─ 厨余垃圾_冰糖葫芦
├─ 厨余垃圾_咖啡
├─ 厨余垃圾_圣女果
├─ 厨余垃圾_地瓜
├─ 厨余垃圾_坚果
├─ 厨余垃圾_壳
├─ 厨余垃圾_巧克力
├─ 厨余垃圾_果冻
├─ 厨余垃圾_果皮
├─ 厨余垃圾_核桃
├─ 厨余垃圾_梨
├─ 厨余垃圾_橙子
├─ 厨余垃圾_残渣剩饭
├─ 厨余垃圾_水果
├─ 厨余垃圾_泡菜
├─ 厨余垃圾_火腿
├─ 厨余垃圾_火龙果
├─ 厨余垃圾_烤鸡
├─ 厨余垃圾_瓜子
├─ 厨余垃圾_甘蔗
├─ 厨余垃圾_番茄
├─ 厨余垃圾_秸秆杯
├─ 厨余垃圾_秸秆碗
├─ 厨余垃圾_粉条
├─ 厨余垃圾_肉类
├─ 厨余垃圾_肠
├─ 厨余垃圾_苹果
├─ 厨余垃圾_茶叶
├─ 厨余垃圾_草莓
├─ 厨余垃圾_菠萝
├─ 厨余垃圾_菠萝蜜
├─ 厨余垃圾_萝卜
├─ 厨余垃圾_蒜
├─ 厨余垃圾_蔬菜
├─ 厨余垃圾_薯条
├─ 厨余垃圾_薯片
├─ 厨余垃圾_蘑菇
├─ 厨余垃圾_蛋
├─ 厨余垃圾_蛋挞
├─ 厨余垃圾_蛋糕
├─ 厨余垃圾_豆
├─ 厨余垃圾_豆腐
├─ 厨余垃圾_辣椒
├─ 厨余垃圾_面包
├─ 厨余垃圾_饼干
├─ 厨余垃圾_鸡翅
├─ 可回收物_不锈钢制品
├─ 可回收物_乒乓球拍
├─ 可回收物_书
├─ 可回收物_体重秤
├─ 可回收物_保温杯
├─ 可回收物_保鲜膜内芯
├─ 可回收物_信封
├─ 可回收物_充电头
├─ 可回收物_充电宝
├─ 可回收物_充电牙刷
├─ 可回收物_充电线
├─ 可回收物_凳子
├─ 可回收物_刀
├─ 可回收物_包
├─ 可回收物_单车
├─ 可回收物_卡
├─ 可回收物_台灯
├─ 可回收物_吊牌
├─ 可回收物_吹风机
├─ 可回收物_呼啦圈
├─ 可回收物_地球仪
├─ 可回收物_地铁票
├─ 可回收物_垫子
├─ 可回收物_塑料制品
├─ 可回收物_太阳能热水器
├─ 可回收物_奶粉桶
├─ 可回收物_尺子
├─ 可回收物_尼龙绳
├─ 可回收物_布制品
├─ 可回收物_帽子
├─ 可回收物_手机
├─ 可回收物_手电筒
├─ 可回收物_手表
├─ 可回收物_手链
├─ 可回收物_打包绳
├─ 可回收物_打印机
├─ 可回收物_打气筒
├─ 可回收物_扫地机器人
├─ 可回收物_护肤品空瓶
├─ 可回收物_拉杆箱
├─ 可回收物_拖鞋
├─ 可回收物_插线板
├─ 可回收物_搓衣板
├─ 可回收物_收音机
├─ 可回收物_放大镜
├─ 可回收物_日历
├─ 可回收物_暖宝宝
├─ 可回收物_望远镜
├─ 可回收物_木制切菜板
├─ 可回收物_木桶
├─ 可回收物_木棍
├─ 可回收物_木质梳子
├─ 可回收物_木质锅铲
├─ 可回收物_木雕
├─ 可回收物_枕头
├─ 可回收物_果冻杯
├─ 可回收物_桌子
├─ 可回收物_棋子
├─ 可回收物_模具
├─ 可回收物_毯子
├─ 可回收物_水壶
├─ 可回收物_水杯
├─ 可回收物_沙发
├─ 可回收物_泡沫板
├─ 可回收物_灭火器
├─ 可回收物_灯罩
├─ 可回收物_烟灰缸
├─ 可回收物_热水瓶
├─ 可回收物_燃气灶
├─ 可回收物_燃气瓶
├─ 可回收物_玩具
├─ 可回收物_玻璃制品
├─ 可回收物_玻璃器皿
├─ 可回收物_玻璃壶
├─ 可回收物_玻璃球
├─ 可回收物_瑜伽球
├─ 可回收物_电动剃须刀
├─ 可回收物_电动卷发棒
├─ 可回收物_电子秤
├─ 可回收物_电熨斗
├─ 可回收物_电磁炉
├─ 可回收物_电脑屏幕
├─ 可回收物_电视机
├─ 可回收物_电话
├─ 可回收物_电路板
├─ 可回收物_电风扇
├─ 可回收物_电饭煲
├─ 可回收物_登机牌
├─ 可回收物_盒子
├─ 可回收物_盖子
├─ 可回收物_盘子
├─ 可回收物_碗
├─ 可回收物_磁铁
├─ 可回收物_空气净化器
├─ 可回收物_空气加湿器
├─ 可回收物_笼子
├─ 可回收物_箱子
├─ 可回收物_纸制品
├─ 可回收物_纸牌
├─ 可回收物_罐子
├─ 可回收物_网卡
├─ 可回收物_耳套
├─ 可回收物_耳机
├─ 可回收物_衣架
├─ 可回收物_袋子
├─ 可回收物_袜子
├─ 可回收物_裙子
├─ 可回收物_裤子
├─ 可回收物_计算器
├─ 可回收物_订书机
├─ 可回收物_话筒
├─ 可回收物_豆浆机
├─ 可回收物_路由器
├─ 可回收物_轮胎
├─ 可回收物_过滤网
├─ 可回收物_遥控器
├─ 可回收物_量杯
├─ 可回收物_金属制品
├─ 可回收物_钉子
├─ 可回收物_钥匙
├─ 可回收物_铁丝球
├─ 可回收物_铅球
├─ 可回收物_铝制用品
├─ 可回收物_锅
├─ 可回收物_锅盖
├─ 可回收物_键盘
├─ 可回收物_镊子
├─ 可回收物_闹铃
├─ 可回收物_雨伞
├─ 可回收物_鞋
├─ 可回收物_音响
├─ 可回收物_餐具
├─ 可回收物_餐垫
├─ 可回收物_饰品
├─ 可回收物_鱼缸
├─ 可回收物_鼠标
├─ 有害垃圾_指甲油
├─ 有害垃圾_杀虫剂
├─ 有害垃圾_温度计
├─ 有害垃圾_灯
├─ 有害垃圾_电池
├─ 有害垃圾_电池板
├─ 有害垃圾_纽扣电池
├─ 有害垃圾_胶水
├─ 有害垃圾_药品包装
├─ 有害垃圾_药片
├─ 有害垃圾_药瓶
├─ 有害垃圾_药膏
├─ 有害垃圾_蓄电池
└─ 有害垃圾_血压计
```

数据集下载之后即可使用，不需要进行进一步的处理，大家只需要记住数据集的位置即可，在后面我们将会把代码中的数据集位置修改为我们自己的数据集位置，比如我这里的数据集位置是 `F:\datas\tmp\data\tttt\trash_jpg` 。

### 模型训练

代码可以在下方的码云链接中进行下载，如果不像训练的小伙伴可以在下方的模型和数据集下载地址下载我训练好的模型，模型的位置了code目录的models目录下。

> 数据集和模型地址： [垃圾分类数据集和tf代码-8w张图片245个类.zip-深度学习文档类资源-CSDN下载](https://download.csdn.net/download/ECHOSON/19713816) 

 ![image-20210625135712173](https://i-blog.csdnimg.cn/blog_migrate/b41b8eac145505b36fa4947797da6d8f.png)

两个模型训练的代码分别是 `train_cnn.py` 和 `train_mobilenet.py` ，分别对应cnn模型和mobilenet模型的训练。

 ![image-20210625135831675](https://i-blog.csdnimg.cn/blog_migrate/970ac75afdc984d9d067f9ad74eea566.png)

#### cnn模型训练

cnn模型训练部分通过tensorflow构建了一个简单的卷积神经网络，这个神经网络由两层卷积和两层池化所构成，之后将特征拉直之后送入到两个全连接层进行分类，模型的具体结构如下：

 ![image-20210625140010130](https://i-blog.csdnimg.cn/blog_migrate/2790d6b77307748c1b7bc3b71f048c20.png)

模型的具体构建如下，这里我们采用adam的优化器，并使用交叉熵作为我们的损失函数。

```python
# 模型加载
def model_load(IMG_SHAPE=(224, 224, 3), class_num=245):
    # 通过keras构建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),  # 归一化，将像素值处理成0到1之间的值
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 卷积层，32个输出通道，3*3的卷积核，激活函数为relu
        tf.keras.layers.MaxPooling2D(2, 2),  # 池化层，特征图大小减半
        # Add another convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层，32个输出通道，3*3的卷积核，激活函数为relu
        tf.keras.layers.MaxPooling2D(2, 2),  # 池化层，特征图大小减半
        tf.keras.layers.Flatten(),  # 将二维的特征图拉直
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),  # 128个神经元的全连接层
        tf.keras.layers.Dense(class_num, activation='softmax')  # 输出层，对应数据集具体的类别数目
    ])
    model.summary()  # 输出模型信息
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])  # 编译模型，指定模型的优化器是adam，模型使用的损失函数的交叉熵损失函数
    return model  # 返回模型
```

如果想要自己训练，只需配置到基本的环境之后，修改数据集的位置即可，我在代码中标注了todo的地方都是大家需要修改的地方，如下图所示，在代码的82行我指明的是我们的数据集的位置，在代码的87行，我指明训练好的模型将会保存在models目录下，并且在代码的95行，我指明了我训练的轮数是30轮，也就是在这8个g的数据集上要跑30个来回。

 ![image-20210625140413822](https://i-blog.csdnimg.cn/blog_migrate/c15b96b78d7e377c5239ee5605570573.png)

训练结束之后，将会得到训练生成的模型保存在models目录下以及训练过程的折线图，在results目录下，如下图所示，从图中可以看出，随着训练的进行，模型的准确率在不断提高，误差在不断下降。

 ![results_cnn_epoch30](https://i-blog.csdnimg.cn/blog_migrate/0b9567bd041a8b5c9a8623c48a5f29ce.png)

#### mobilenet模型训练

mobilenet是我非常喜欢的一个模型，这个模型采用了深度可分离卷积的技术，提高了网络的层次，提升了准确率，而且模型非常轻量，训练好的mobilenet模型不到20m，在cpu的条件下或者是在手机的条件下都有不错的性能，模型的结构图如下，模型比较深，我只截取了部分，大家可以下载 `netron` 软件自行查看模型结构。

 ![image-20210625142218267](https://i-blog.csdnimg.cn/blog_migrate/6e576a0c96e67a6fc2792a32bb9e0f6f.png)

这里的训练方式我们采用的是迁移学习的方式，即只使用mobilenet的特征提取层，特征提取层的参数已经在大型数据集imagenet上训练好，不需要再进行训练，我们只需要添加全连接层，对应到具体的分类数据即可，详细的代码如下：

```python
# 模型加载，指定图片处理的大小和是否进行迁移学习
def model_load(IMG_SHAPE=(224, 224, 3), class_num=245):
    # 微调的过程中不需要进行归一化的处理
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet') # 加载mobilenetv2模型
    base_model.trainable = False # 将主干的特征提取网络参数进行冻结
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE), # 归一化处理，将像素值处理为-1到1之间
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), # 全局平均池化
        tf.keras.layers.Dense(class_num, activation='softmax') # 设置最后的全连接层，用于分类
    ])
    model.summary() # 输出模型信息
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # 用于编译模型，指定模型的优化器是adam优化器，模型的损失函数是交叉熵损失函数
    return model # 返回模型
```

同cnn模型的训练一样，如果需要自己训练的话，只需要修改几个参数即可，如下所示：

 ![image-20210625142755881](https://i-blog.csdnimg.cn/blog_migrate/7709e2a291b04e5dad7e9a527a53a297.png)

训练完成之后，同样是得到训练好的模型，模型将会保存再models目录下，以及训练过程中的折线图，如下所示：

 ![results_mobilenet_epoch30](https://i-blog.csdnimg.cn/blog_migrate/4d2056ac7f37b83615059297c13c497c.png)

### 模型测试

模型的测试部分也是分别对两个模型进行测试，主要是测试两个模型在测试集上的准确率表现，测试的代码是 `test_model.py` ，模型的测试通过evaluate函数进行，执行该函数之前需要先加载数据集和训练好的模型，我们有两个模型，所以我们应该要得到这两个模型的准确率信息，需要修改的地方有四处，如下所示：

 ![image-20210625143505273](https://i-blog.csdnimg.cn/blog_migrate/ff3d9a9a28781f26a0914e43ac4b72ad.png)

修改完成之后直接执行该文件即可，会在命令行中输出具体的准确率信息，如下所示：

 ![image-20210625143818024](https://i-blog.csdnimg.cn/blog_migrate/09a89b20d7b51484d4b20b93438ad6eb.png)

上面两个红框就是模型对应的具体准确率信息，可以看出cnn模型的准确率只有68%，mobilnet的准确率比较高，大概是在83%左右，也就是相对于cnn模型来说，mobilenet模型的准确率大概有15个百分点的提升，所以在后面的模型使用的时候我们选择mobilnet模型。

另外，我还对mobilenet模型测试了四个大类上的分别准确率信息，由于数据集过大，这部分代码在跑的时候十分不稳定，所以在这里我放了结果，具体的逻辑也在test_model代码中，需要现将测试结果保存在pkl文件之后再进行热力图的绘制，具体的结果如下：

 ![测试热力图](https://i-blog.csdnimg.cn/blog_migrate/4fcc2c93e87737fa9ba86d2cf9ced3ea.png)

### 模型使用

#### 桌面程序构建

桌面程序的构建使用的是pyqt5，pyqt5构建出来的界面相对比较美观一些，大家只需要掌握qt中的信号和槽函的机制就能很方便的开发出自己的界面程序，我们所开发的界面程序主要是两个功能，即上传图片和图片预测的功能，界面的程序是代码中的 `window_trash.py` ，大家需要在代码中的18行修改为自己训练好的模型位置即可。

 ![image-20210625150201587](https://i-blog.csdnimg.cn/blog_migrate/5727b4b4821561ed449447492b7fdff1.png)

其中图片上传和预测的逻辑如下，用户点击上传图片的功能进行图片的上传，系统将图片处理成224的大小方便后面送入模型进行预测，点击预测按钮之后系统调用在18行中加载好的模型预测结果并进行显示。

```python
    # 上传图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '', 'Image files(*.jpg *.png *jpeg)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            target_image_name = "images/tmp_single" + img_name.split(".")[-1]
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))

    # 预测图片
    def predict_img(self):
        img = Image.open('images/target.png')
        img = np.asarray(img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]
        names = result.split("_")
        # print(result)
        if names[0] == "厨余垃圾":
            self.label_info.setText(
                "厨余垃圾是指居民日常生活及食品加工、饮食服务、单位供餐等活动中产生的垃圾，包括丢弃不用的菜叶、剩菜、剩饭、果皮、蛋壳、茶渣、骨头等。由于厨余垃圾含有极高的水分与有机物，很容易腐坏，产生恶臭。经过妥善处理和加工，可转化为新的资源，高有机物含量的特点使其经过严格处理后可作为肥料、饲料，也可产生沼气用作燃料或发电，油脂部分则可用于制备生物燃料。")
        if names[0] == "有害垃圾":
            self.label_info.setText(
                "有害垃圾指对人体健康或者自然环境造成直接或者潜在危害的生活废弃物。常见的有害垃圾包括废灯管、废油漆、杀虫剂、废弃化妆品、过期药品、废电池、废灯泡、废水银温度计等，有害垃圾需按照特殊正确的方法安全处理，一般需要经过特殊的处理之后才可以进行焚烧，堆肥，填埋处理")
        if names[0] == "可回收物":
            self.label_info.setText(
                " 根据《城市生活垃圾分类及其评价标准》行业标准以及参考德国垃圾分类法，可回收物是指适宜回收循环使用和资源利用的废物。主要包括：纸类，塑料，金属，玻璃，织物等。主要的处理方式有：1.垃圾再生法；2.垃圾焚烧法；3.垃圾堆肥法；4.垃圾生物降解法。")
        if names[0] == "其他垃圾":
            self.label_info.setText(
                "其他垃圾指危害比较小，没有再次利用的价值的垃圾，其他垃圾包括砖瓦陶瓷、渣土、卫生间废纸、瓷器碎片、动物排泄物、一次性用品等难以回收的废弃物。一般都采取填埋、焚烧、卫生分解等方法处理，部分还可以使用生物分解的方法解决")
        self.result.setText(names[0])
        self.result_f.setText(names[1])
```

效果如下：

 ![image-20210625150904213](https://i-blog.csdnimg.cn/blog_migrate/86f3b77177c9f4074f6fc22b633a5dd5.png)

#### flask网页端程序构建

程序还在构建中，感兴趣的小伙伴可以持续关注，大概7月份会出相关教程

#### 手机安卓程序构建

程序还在构建中，感兴趣的小伙伴可以持续关注，大概7月份会出相关教程

### 写在最后

大家好，这里是dejahu，你所看到的这篇教程是大作业怎么搞系列的其中一篇，大作业怎么搞系列的初衷是为了帮助大家写大作业的时候能够有所参考，目前来看这个系列的反响还不错，这个系列目前出的教程基本都是围绕使用tensorflow2.3建立图像分类模型进行的，关于如何使用tensorflow2.3训练自己的数据集我也出了一期详细的教程（ [手把手教你用tensorflow2.3训练自己的分类数据集_dejahu的博客-CSDN博客](https://blog.csdn.net/ECHOSON/article/details/117964477) ），所以目前关于图像分类的部分我打算暂时就不再更新了，有问题的小伙伴可以在评论区留言。

因为我本身的研究方向是计算机视觉，擅长的开发语言是python，所以下面会有一些其他的策划，大概是这样的：

- 网页端和移动端的模型部署

- 详解卷积神经网络

- 一些好的神经网络模型的介绍

- 使用pytorch进行图像分类

- yolo系列算法

- Pyqt5构建图形化界面教程

- 图像处理基础知识

最后，还是感谢大家一路一来的支持，嘻嘻！