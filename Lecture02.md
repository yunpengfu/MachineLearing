# 第2章 k-临近算法
- k-近邻（kNN, k-NearestNeighbor）算法是一种基本分类与回归方法，我们这里只讨论分类问题中的 k-近邻算法。
- k-近邻算法，采取测量不同特征值之间的距离方法进行分类。一般来说，我们只选择样本数据前k个最相似的数据，选择k个最相似数据中出现次数最多的分类，作为新数据的分类。  
> k近邻算法实际上利用训练数据集对特征向量空间进行划分，并作为其分类的“模型”。 k值的选择、距离度量以及分类决策规则是k近邻算法的三个基本要素。

# KNN 原理
1. 假设有一个带有标签的样本数据集（训练样本集），其中包含每条数据与所属分类的对应关系。
2. 输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较。
   a. 计算新数据与样本数据集中每条数据的距离。
   b. 对求得的所有距离进行排序（从小到大，越小表示越相似）。
   c. 取前 k （k 一般小于等于 20 ）个样本数据对应的分类标签。
3. 求 k 个数据中出现次数最多的分类标签作为新数据的分类。
> KNN 通俗理解：给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的 k 个实例，这 k 个实例的多数属于某个类，就把该输入实例分为这个类
## KNN 算法特点
优点：精度高、对异常值不敏感、无数据输入假定  
缺点：计算复杂度高、空间复杂度高  
适用数据范围：数值型和标称型  
## KNN 开发流程
收集数据：任何方法  
准备数据：距离计算所需要的数值，最好是结构化的数据格式  
分析数据：任何方法  
训练算法：此步骤不适用于 k-近邻算法  
测试算法：计算错误率  
使用算法：输入样本数据和结构化的输出结果，然后运行 k-近邻算法判断输入数据分类属于哪个分类，最后对计算出的分类执行后续处理  
## KNN 伪代码
K-邻居算法的伪代码：
1. 计算已有了类别数据集中的点与当前点之间的距离；
2. 按距离递增次序排序；
3. 选取与当前点距离最小的k个点；
4. 确定前k个点所在类别出现的频率；
5. 返回前k个点出现频率最高的类别作为当前点的预测分类。

# k-临近算法 代码
1. Python导入数据
```
   from numpy import *
   import operator
   # 导入科学计算包numpy和运算符模块operator
   
   def createDataSet():
      group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
      labels = ['A', 'A', 'B', 'B']
      return group, labels
```
createDataSet()函数，创建数据集和标签。  
调用方式：
> import kNN  
> group, labels = kNN.createDataSet()
2. kNN算法
```
   def classify0(inX, dataSet, labels, k):
      # inX 用于分类的输入向量
      # dataSet 输入的样本训练集
      # labels标签向量
      # k 用于选择最近邻居数目
      dataSetSize = dataSet.shape[0]   # ".shape[0]"输出行数,".shape[1]"输出列数
      diffMat = tile(inX, (dataSetSize,1)) - dataSet  # 在列方向上重复inX一次，行dataSetSize次
      """
       diffMat 为欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet的第一个点的距离。
       第二行： 同一个点 到 dataSet的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet的第N个点的距离。
      [ [2,3],[2,3] ]-[ [2,3],[2,0] ]  # [ [x-x1,y-y1],[x-x2,y-y2] ]
      """
      sqDiffMat = diffMat**2  # 取平方 [x-x1,y-y1]^2 + [x-x2,y-y2]^2
      sqDistances = sqDiffMat.sum(axis=1) # 将矩阵的每一行相加
      distances = sqDistances ** 0.5   # 开方 { [x-x1,y-y1]^2 + [x-x2,y-y2]^2 }^0.5
      sortedDistIndicies = distances.argsort()  # 根据距离排序从小到大的排序，返回对应的索引位置
      classCount = {}   # 定义一个元组
      for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] # 找到该样本的类型
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   # 计数
      sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)、
      """
      字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
      例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
      sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
      例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2（即按照的是元组的值）进行排序的，而不是a,b,c
      b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c（即比较的是元组的键）而不是0,1,2
      b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序。
      """
      return sortedClassCount[0][0]      
```

# 应用示例
## 优化约会网站的配对效果
### 项目概述
海伦使用约会网站寻找约会对象。经过一段时间之后，她发现曾交往过三种类型的人:
* 不喜欢的人
* 魅力一般的人
* 极具魅力的人
海伦收集了主要包含以下3种特征的约会数据（保存在datingTestSet2.txt中）：
1. 每年获得的飞行常客里程数
2. 玩视频游戏所耗时间百分比
3. 每周消费的冰淇淋公升数
### 示例：在约会网站上使用k-临近算法
> 收集数据：提供文本文件  
准备数据：使用 Python 解析文本文件  
分析数据：使用 Matplotlib 画二维散点图  
训练算法：此步骤不适用于 k-近邻算法  
测试算法：使用海伦提供的部分数据作为测试样本。  
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。  
使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。
### 代码
1. 使用 Python 解析文本文件，将文本记录转换为 NumPy 的解析程序
```
def file2matrix(filename):
   fr = open(filename)
   numberOfLines = len(fr.readlines()) # 获得文件中的数据行的行数
   returnMat = zeros((numberOfLines, 3))  # 生成对应的空矩阵,例如：zeros(2，3)就是生成一个2*3的矩阵，各个位置上全是0，此矩阵即为返回的矩阵
   classLabelVector = []  # 此矩阵为返回的标签矩阵
   index = 0
   for line in fr.readlines():
       line = line.strip() # str.strip([chars]),返回已移除字符串头尾指定字符所生成的新字符串
       listFromLine = line.split('\t') # 以 '\t' 切割字符串
       ```
         eg.
         txt = "Google#Runoob#Taobao#Facebook"
         x = txt.split("#")
         output:
         ['Google','Runoob','Taobao','Facebook']
       ```
       returnMat[index, :] = listFromLine[0:3]  # 每行的属性数据
       ```
         eg.
         jj = [ [1,2,3],[8,8,8] ]
         jj[1,:]  # 取出jj第1行元素
         output：
         [ [8,8,8] ]
       ```
       classLabelVector.append(int(listFromLine[-1])) # 每列的类别数据，就是 label 标签数据
       index += 1
   return returnMat, classLabelVector  # 返回数据矩阵returnMat和对应的类别classLabelVector
   
```
2. 分析数据：使用 Matplotlib 画二维散点图
## 手写数字识别系统





