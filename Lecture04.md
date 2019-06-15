# 第4章 基于概率论的分类方法：朴素贝叶斯
贝叶斯分类是一类分类算法的总称，这类算法均以贝叶斯定理为基础，故统称为贝叶斯分类。本章首先介绍贝叶斯分类算法的基础——贝叶斯定理。  
* 贝叶斯理论
> 假设我们现在有一个数据集，它由两类数据组成，数据分布如下图4-1所示：
> ![数据分布](https://raw.githubusercontent.com/apachecn/AiLearning/master/img/ml/4.NaiveBayesian/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E7%A4%BA%E4%BE%8B%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%83.png "数据分布")
> 我们现在用 p1(x,y) 表示数据点 (x,y) 属于类别 1（图中用圆点表示的类别）的概率，用 p2(x,y) 表示数据点 (x,y) 属于类别 2（图中三角形表示的类别）的概率，那么对于一个新数据点 (x,y)，可以用下面的规则来判断它的类别：
>   * 如果 p1(x,y) > p2(x,y) ，那么类别为1；
>   * 如果 p2(x,y) > p1(x,y) ，那么类别为2。  

> 也就是说，我们会选择高概率对应的类别。这就是贝叶斯决策理论的核心思想，即选择具有最高概率的决策。
* 条件概率
> 假设现在有一个装了 7 块石头的罐子，其中 3 块是白色的，4 块是黑色的（如图4-2所示）。如果从罐子中随机取出一块石头，那么是白色石头的可能性是多少？由于取石头有 7 种可能，其中 3 种为白色，所以取出白色石头的概率为 3/7 。那么取到黑色石头的概率又是多少呢？很显然，是 4/7 。我们使用 P(white) 来表示取到白色石头的概率，其概率值可以通过白色石头数目除以总的石头数目来得到。  
> ![石头分布](https://raw.githubusercontent.com/apachecn/AiLearning/master/img/ml/4.NaiveBayesian/NB_2.png "石头分布")
> 如果这7块石头如图4-3所示，放在两个桶中，那么上述概率应该如何计算？ 
> ![石头分布2](https://raw.githubusercontent.com/apachecn/AiLearning/master/img/ml/4.NaiveBayesian/NB_3.png "石头分布2")
> 计算 P(white) 或者 P(black) ，如果事先我们知道石头所在桶的信息是会改变结果的。这就是所谓的条件概率（conditional probablity）。假定计算的是从 B 桶取到白色石头的概率，这个概率可以记作 P(white|bucketB) ，我们称之为“在已知石头出自 B 桶的条件下，取出白色石头的概率”。很容易得到，P(white|bucketA) 值为 2/4 ，P(white|bucketB) 的值为 1/3 。  
> 条件概率的计算公式如下：
   > P(white|bucketB) = P(white and bucketB) / P(bucketB)
> 首先，我们用 B 桶中白色石头的个数除以两个桶中总的石头数，得到 P(white and bucketB) = 1/7 .其次，由于 B 桶中有 3 块石头，而总石头数为 7 ，于是 P(bucketB) 就等于 3/7 。于是又 P(white|bucketB) = P(white and bucketB) / P(bucketB) = (1/7) / (3/7) = 1/3 。  
> 另外一种有效计算条件概率的方法称为贝叶斯准则。贝叶斯准则告诉我们如何交换条件概率中的条件与结果，即如果已知 P(x|c)，要求 P(c|x)，那么可以使用下面的计算方法：
