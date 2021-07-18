# https: // www.e - learn.cn / content / python / 2198918
import sklearn
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


# 数据预处理过滤式特征选取SelectKBest模型
def test_SelectKBest():
    # X = [[1, 2, 3, 4, 5],
    #      [3, 3, 3, 3, 3],
    #      [1, 1, 1, 1, 1]]

    # 读取数据

    data = pd.read_csv("DrDoS_NTP.csv", low_memory=False)
    # data = pd.DataFrame(data, dtype='float')

    from sklearn.feature_extraction import DictVectorizer
    import csv
    from sklearn import tree
    from sklearn import preprocessing
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    # from sklearn.externals.six import StringIO

    # sklearn对数据有格式要求，首先要对数据进行格式预处理。
    # Read in the csv file and put features into list of dict and list of class label
    # 读取csv文件，并把属性放到字典列表和类标签中
    # Python2.x
    # allElectronicsData = open(r'AllElectronics.csv', 'rb')
    # reader = csv.reader(allElectronicsData)
    # headers = reader.next()
    # 上面的语句在python3.X会报错，'_csv.reader' object has no attribute 'next'
    # 在python3.x需改为如下语句
    allElectronicsData = open(r'test1.csv', 'rt')
    # allElectronicsData = open(r'test.csv', encoding = "gb18030", errors = "ignore")

    reader = csv.reader(allElectronicsData)
    headers = next(reader)

    print(headers)

    featureList = []
    labelList = []

    for row in reader:
        labelList.append(row[len(row) - 1])
        rowDict = {}
        for i in range(1, len(row) - 1):
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)

    print(featureList)

    # 从表中可以看出是用字典储存，所以是无序的。

    # Vetorize features
    vec = DictVectorizer()
    X = vec.fit_transform(featureList).toarray()

    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(labelList)


    # data = pd.read_csv("DrDoS_NTP.csv", header=None,low_memory=False)

    # X = data.iloc[:, 0:85]
    # X = np.array(X)
    # y = data.iloc[:, -1:]
    # y = np.array(y)
    # y = np.delete(y,1,0);
    print(X);
    print(X.size)
    print(y);
    print(y.size)

    # sklearn.feature_selection.chi2(X, y)
    #
    #
    # y = [0, 1, 0, 1]
    # print("before transform:", X)
    selector = SelectKBest(score_func=f_classif, k=25) .fit_transform(X, y.ravel())
    # selector.fit(X, y)
    selector.scores_ = selector.scores_.tolist()
    selector.pvalues_ = selector.pvalues_.tolist()
    print("scores_:", selector.scores_)
    print("pvalues_:", selector.pvalues_)
    print("selected index:", selector.get_support(True))
    print("after transform:", selector.transform(X))


# 调用test_SelectKBest()
test_SelectKBest()