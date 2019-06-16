# little conv augmented



![1559997914636](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1559997914636.png)



## Epoch 1

![1559999044215](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1559999044215.png)

![1559999016090](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1559999016090.png)

![1559999028887](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1559999028887.png)

## Epoch15

![1559999134961](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1559999134961.png)

![1559999125152](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1559999125152.png)

![1559999113627](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1559999113627.png)





## Transfer Learning 

Using resnet16 , and replace the last FC layer

![img](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\transferleraning.png)

### The attention of the model 

#### epoch 0

![1560063627272](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1560063627272.png)

#### epoch 10

![1560063673428](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1560063673428.png)

#### epoch20

![1560063924243](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1560063924243.png)

可以看到神经网络的注意力经过训练放到了狗脸上。

另外，再到后面gradients开始出现大量全都是零的情况，导致grad cam无法使用，具体是什么原因导致gradients出现全零还没有搞清楚。



## 理解resnet16如何看图片（对resnet最后一个卷积层做tsne）

选取最后一个卷积层的average pooling结果作为embedding 向量。

选择Perplexity=20，二维聚类效果如下：

![1560065480144](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1560065480144.png)

其中下面孤立部分大部分由比较同一种狗组成，如图：

![1560065215036](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1560065215036.png)

在各个局部，也可以看到，神经网络基本上按照狗的不同形态特征进行区分。比如下图可以看到黑白相间和金色毛茸茸分成了两个部分

![1560065300877](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1560065300877.png)



所以可以看出，经过预训练的resnet可以区分不同品种的狗，关注到狗本身的差异，而不是按照狗的姿势，背景等等区分。对比从头开始训练的网络的embedding结果，可以解释transfer效果好的原因。





### 对比没有使用transfer learning的模型

下图是对没有使用transfer learn而从头训练的小网络的最后一层进行PCA的结果，对可以从下图中看到，图像被按照颜色分为了左上右下两个部分，而构成主要颜色的是图片的背景，与图片中的狗的品种无关。

![1560069862056](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1560069862056.png)



而对这个模型的最后一层特征做t-sne，得到的结果如下（perplexity = 32）

仍然可以看到，基本上按照颜色聚的类

![1560070504170](D:\yangcy\UNVjunior\CS385\PROJ2\CS385ML\classification\docs\pics\1560070504170.png)

细看之下可以发现，聚类的因素和狗的品种没有关系。

