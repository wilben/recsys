# recsys
融合评论文本的推荐模型

![image](https://github.com/blazewint3r/recsys/blob/master/%E5%9B%BE%E7%89%871.png)

上图是本文提出的深度网络的简单示意图。

对评分部分，采用[1]提出的latent facor模型。对评论部分，采用[2]中提出的doc2vec模型建模。并希望由此约束隐藏层的user/item lantent facor的取值，提高评分预测的准确性。

为简单起见，上图仅表示了 “一条评论中的一个大小为t个词的窗口”作为一个输入样本的情形。输入的是这条评论的用户、商品、以及窗口中t个词。输出是窗口中心词和该条评论对应的评分。

其中U和I分别代表商品和评论的one hot表示，它们经过线性映射层，被映射为User Embedding和Item Embedding(简写为U_eb和I_eb). 数个W代表窗口中各个词的one hot表示，它们同样经过线性映射，映射为word Embedding. (简写为W_eb)

User Embedding 和Item Embedding经过concatenate作为非线性层的输入，变换为Document Embedding（简写为D_eb)（D_eb和[2]中定义的Document Embedding具有相同的语义），D_eb和窗口中各个词的W_eb进行连接，输出经过一个softmax分类器，对窗口中心词进行预测。

User Embedding 和Item Embedding同时也各自经过另一个非线性层，分别变换为User Latent Factor和Item Latent Factor（这和Yehuda Koren等人在[1]中定义的Latent factor具有相同的语义），它们的内积，加上用户和商品各自对应的bias以及全局bias（图中简单起见未画出），输出即为评分的预测。

模型的损失函数为两部分的预测输出的与真实输出的交叉熵以及网络正则项的加权和（权重为模型的超参数之一）。

训练中，针对Latent Factor部分非凸，采用交替梯度下降+反向传播算法求解参数。

  [1] Koren Y, Bell R, Volinsky C. Matrix Factorization Techniques for Recommender Systems[J]. Computer, 2009, 42(8):30-37.
  [2] Tomas Mikolov, Ilya Sutskeve, Kai Chen, Greg Corrado, and Jeffrey Dean. 2012. Distributed
  ​
