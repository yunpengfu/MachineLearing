# 第4章 基于概率论的分类方法：朴素贝叶斯
贝叶斯分类是一类分类算法的总称，这类算法均以贝叶斯定理为基础，故统称为贝叶斯分类。本章首先介绍贝叶斯分类算法的基础——贝叶斯定理。  
* 贝叶斯理论
>假设我们现在有一个数据集，它由两类数据组成，数据分布如下图4-1所示：
>![数据分布](https://raw.githubusercontent.com/apachecn/AiLearning/master/img/ml/4.NaiveBayesian/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF%E7%A4%BA%E4%BE%8B%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%83.png "数据分布")  
>我们现在用 p1(x,y) 表示数据点 (x,y) 属于类别 1（图中用圆点表示的类别）的概率，用 p2(x,y) 表示数据点 (x,y) 属于类别2图中三角形表示的类别）的概率，那么对于一个新数据点 (x,y)，可以用下面的规则来判断它的类别：
>* 如果 p1(x,y) > p2(x,y) ，那么类别为1；
>* 如果 p2(x,y) > p1(x,y) ，那么类别为2。  

>也就是说，我们会选择高概率对应的类别。这就是贝叶斯决策理论的核心思想，即选择具有最高概率的决策。
* 条件概率
>假设现在有一个装了7块石头的罐子，其中3块是白色的，4块是黑色的（如图4-2所示）。如果从罐子中随机取出一块石头，那么是白色石头的可能性是多少？由于取石头有7种可能，其中3种为白色，所以取出白色石头的概率为3/7 。那么取到黑色石头的概率又是多少呢？很显然，是4/7 。我们使用P(white)来表示取到白色石头的概率，其概率值可以通过白色石头数目除以总的石头数目来得到。  
>![石头分布](https://raw.githubusercontent.com/apachecn/AiLearning/master/img/ml/4.NaiveBayesian/NB_2.png "石头分布")  
>如果这7块石头如图4-3所示，放在两个桶中，那么上述概率应该如何计算？ 
>![石头分布2](https://raw.githubusercontent.com/apachecn/AiLearning/master/img/ml/4.NaiveBayesian/NB_3.png "石头分布2")  
>计算 P(white) 或者 P(black) ，如果事先我们知道石头所在桶的信息是会改变结果的。这就是所谓的条件概率（conditional probablity）。假定计算的是从B桶取到白色石头的概率，这个概率可以记作 P(white|bucketB) ，我们称之为“在已知石头出自B桶的条件下，取出白色石头的概率”。很容易得到，P(white|bucketA) 值为2/4 ，P(white|bucketB) 的值为1/3 。  
>条件概率的计算公式如下：
>>P(white|bucketB) = P(white and bucketB) / P(bucketB)  

>首先，我们用B桶中白色石头的个数除以两个桶中总的石头数，得到P(white and bucketB) = 1/7。其次，由 B桶中有3块石头，而总石头数为 7 ，于是 P(bucketB) 就等于3/7 。于是又 P(white|bucketB) = P(white and bucketB) / P(bucketB) = (1/7) / (3/7) = 1/3 。  
>另外一种有效计算条件概率的方法称为贝叶斯准则。贝叶斯准则告诉我们如何交换条件概率中的条件与结果，即如果已知P(x|c)，要求P(c|x)，那么可以使用下面的计算方法：
>>![计算方法](https://raw.githubusercontent.com/apachecn/AiLearning/master/img/ml/4.NaiveBayesian/NB_4.png "计算方法")  

## 朴素贝叶斯 原理
```
提取所有文档中的词条并进行去重
获取文档的所有类别
计算每个类别中的文档数目
对每篇训练文档: 
    对每个类别: 
        如果词条出现在文档中-->增加该词条的计数值（for循环或者矩阵相加）
        增加所有词条的计数值（此类别下词条总数）
对每个类别: 
    对每个词条: 
        将该词条的数目除以总词条数目得到的条件概率（P(词条|类别)）
返回该文档属于每个类别的条件概率（P(类别|文档的所有词条)）
```
### 朴素贝叶斯 算法特点
优点: 在数据较少的情况下仍然有效，可以处理多类别问题。  
缺点: 对于输入数据的准备方式较为敏感。  
适用数据类型: 标称型数据。  
### 朴素贝叶斯 开发流程
收集数据: 可以使用任何方法。  
准备数据: 需要数值型或者布尔型数据。  
分析数据: 有大量特征时，绘制特征作用不大，此时使用直方图效果更好。  
训练算法: 计算不同的独立特征的条件概率。  
测试算法: 计算错误率。  
使用算法: 一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定非要是文本。  

## 朴素贝叶斯算法 代码
1. 词表到向量的转换：
```
  def loadDataSet():    # 创建数据集
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], #[0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec
    
  def createVocabList(dataSet):     # 用set数据类型，创建不包含重复词的列表
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符 | 用于求两个集合的并集
    return list(vocabSet)
    
  def setOfWords2Vec(vocabList, inputSet):
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocabList)   # [0,0......]
    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
    
  def bagOfWords2VecMN(vocabList, inputSet):    # 朴素贝叶斯词袋模型
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
```
2. 朴素贝叶斯分类器训练函数trainNB0()：
```
  def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix) # 总文件数
    numWords = len(trainMatrix[0])  # 总单词数
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 侮辱性文件的出现概率
    # 避免单词列表中的任何一个单词为0，而导致最后的乘积为0，所以将每个单词的出现次数初始化为 1
    p0Num = ones(numWords)  # 正常的统计，[0,0......]->[1,1,1,1,1.....]
    p1Num = ones(numWords)  # 侮辱的统计
    # 整个数据集单词出现总数，2.0根据样本/实际调查结果调整分母的值（2主要是避免分母为0，当然值可以调整
    p0Denom = 2.0   # 正常的统计
    p1Denom = 2.0   # 侮辱的统计
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] # 累加辱骂词的频次
            p1Denom += sum(trainMatrix[i])  # 对每篇文章的辱骂的频次 进行统计汇总
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    p1Vect = log(p1Num / p1Denom)
    # 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive
```
3. 朴素贝叶斯分类函数：
```
  def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
    return 0
  def testingNB():
    listOPosts, listClasses = loadDataSet() # 1. 加载数据集
    myVocabList = createVocabList(listOPosts)   # 2. 创建单词集合
    trainMat = []   # 3. 计算单词是否出现并创建数据矩阵
    for postinDoc in listOPosts:    # 返回m*len(myVocabList)的矩阵， 记录的都是0，1信息
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))   # 4. 训练数据
    testEntry = ['love', 'my', 'dalmation'] # 5. 测试数据
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
```

## 应用示例
### 使用朴素贝叶斯过滤垃圾邮件
邮件在'MCIA_Code/Ch04/email/'文件夹中：
```
  def textParse(bigString):
    import re
    # 使用正则表达式来切分句子，其中分隔符是除单词、数字外的任意字符串
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
    
  def spamTest():   # 对贝叶斯垃圾邮件分类器进行自动化处理。
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):      # 切分，解析数据，并归类为 1 类别
        wordList = textParse(open('MCIA_Code/Ch04/email/pam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)
        # 切分，解析数据，并归类为 0 类别
        wordList = textParse(open('MCIA_Code/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)    
    vocabList = createVocabList(docList)    # 创建词汇表
    trainingSet = range(50)
    testSet = []
    for i in range(10):     # 随机取 10 个邮件用来测试
        # random.uniform(x, y) 随机生成一个范围为 x - y 的实数
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the errorCount is: ', errorCount)
    print('the testSet length is :', len(testSet))
    print('the error rate is :', float(errorCount)/len(testSet))
```
### 使用朴素贝叶斯分类器从个人广告中获取区域倾向
```
  def calcMostFreq(vocabList,fullText):     #RSS源分类器及高频词去除函数
    import operator
    freqDict={}
    for token in vocabList:  #遍历词汇表中的每个词
        freqDict[token]=fullText.count(token)  #统计每个词在文本中出现的次数
    sortedFreq=sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)  #根据每个词出现的次数从高到底对字典进行排序
    return sortedFreq[:30]   #返回出现次数最高的30个单词
    
    
  def localWords(feed1,feed0):
    import feedparser
    docList=[];classList=[];fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])   #每次访问一条RSS源
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:vocabList.remove(pairW[0])    #去掉出现次数最高的那些词
    trainingSet=range(2*minLen);testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is:',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V


  def getTopWords(ny,sf):       # 最具表征性的词汇显示函数
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0:topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0:topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
    print(item[0])
```
