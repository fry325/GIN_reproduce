# GIN_reproduce
reproduce GNN model: GIN (XU et al. 2019)  
通过复现《How powerful are graph neural networks?》这篇论文，熟悉图分类常用数据集。必须注意到cora, citeseer和pubmed这三个数据集不是用于graph分类的，而是用于node分类的。  
GNN常用的公开数据集大致有分子、蛋白质、论文引用网络、社交网络等这几类。  
官方[代码地址](https://github.com/weihua916/powerful-gnns)  
本人复现[代码地址](https://github.com/fry325/GIN_reproduce)  
[常用图数据集](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)  
吐槽：为什么每个很牛的模型，开源的数据预处理代码写得都和翔一样？  
吐槽：编写图神经网络模型，我感觉构造coo格式的稀疏矩阵才是最头疼的
# 数据集描述
1. IMDB数据集：每个节点代表一个演员，每条边代表两个演员是否出现在同一个电影里。IMDB-BINARY类别有两个：爱情片和动作片（？？？）如果同时是爱情片和动作片的话，就会归类为动作片。IMDB-MULTI则在爱情片和动作片的基础上，加了一个科幻片类别。
2. COLLAB数据集：每个节点代表一个researcher，每个graph我猜是代表了一个科研团体？每一个Graph有一个类别，共有高能物理、凝聚态物理和天体物理3个类别。
3. Reddit数据集：每个节点代表一个用户，每个graph代表一个帖子，帖子里如果一个用户回复了另一个用户的评论就构成一条边。graph的类别代表了帖子类型。Reddit-Binary、Reddit5K、Reddit12K分别有2类、2类和11类。
4. MUTAG数据集：每个graph代表一个硝基化合物分子，有两个类别，代表这个分子是诱变芳香族或杂芳香族。
5. PTC数据集：也是分子化合物，和上述一样
6. NCI1数据集：也是分子化合物，和上述一样
7. Protein数据集：每个节点是一个secondary structure elements，如果两个节点在氨基酸序列或3D空间中是相邻节点就会存在一条边。
总结如下表

| 数据集  | 图数量 | 图类别数 | 图平均节点 | 节点标签数 |
| ----  | ----  | ----  | ----  | ----  |
| IMDB-binary | 1000  |   2  |   19.77  |  0  |  
| IMDB-multi | 1500  |   3  |   13  |  0  |  
| COLLAB | 5000  |   3  |   74.49  |  0  |  
| Reddit-Binary | 2000  |   2  |  429.61  |  0  |  
| Reddit-multi-5K | 5000  |   2  |   508.5  |  0  |  
| Reddit-multi-12K | 11929 |   11  |   391.4  |  0  |  
| MUTAG | 188 |   2  |   17.9  |  7  |  
| PTC | 344 |   2  |   25.5  |  19  |  
| NCI1 | 4110 |   2  |   29.8  |  37  |  
| Protein | 1113 |   2  |   39.1  |  3  |  

# Graph Isomorphism Network复现结果
论文给的结果如下图，这个结果好像是跑了10 folds，把test set的ACC取平均的结果。精力有限，我只跑了一个fold。官方的GIN模型code有个细节，就是参考了Xu et al. 2018提出的JKNet，使用了Layer Aggregator。每一层的输出，都经过一个Linear层，再求和，作为最后的输出。
![GIN官方实验结果](https://upload-images.jianshu.io/upload_images/21290480-9d544eba18bf6bcd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我复现的结果见以下7张图。模型设置：neighbor aggregator和graph aggregator都使用sumPool，没加epsilon参数（也就是论文的GIN0，而不是GIN-eps）。由于资源有限，Reddit数据集跑起来OOM了，就不跑Reddit的了。
如图所示，Test Accuracy在某个Epoch都刚好达到了文中给的结果，NCI1数据集除外，我也不知道为什么。
![IMDB_BIN_GIN0](https://upload-images.jianshu.io/upload_images/21290480-f2a22c01f36f0dfb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![IMDB_MULTI_GIN0](https://upload-images.jianshu.io/upload_images/21290480-fa71b52c4c27128c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![MUTAG_GIN0](https://upload-images.jianshu.io/upload_images/21290480-dec98da6ef688909.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![NCI1_GIN0](https://upload-images.jianshu.io/upload_images/21290480-cd5a9151a8d32552.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![PROTEINS_GIN0](https://upload-images.jianshu.io/upload_images/21290480-a15b5feb8a6b96f1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![PTC_GIN0](https://upload-images.jianshu.io/upload_images/21290480-81c28a07bf70d75c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![COLLAB_GIN0](https://upload-images.jianshu.io/upload_images/21290480-9ecaf286f43099dd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
