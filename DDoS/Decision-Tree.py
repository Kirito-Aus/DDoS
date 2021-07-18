# from sklearn.feature_extraction import DictVectorizer
# import csv
# from sklearn import tree
# from sklearn import preprocessing
# # from sklearn.externals.six import StringIO
#
# #sklearn对数据有格式要求，首先要对数据进行格式预处理。
# # Read in the csv file and put features into list of dict and list of class label
# #读取csv文件，并把属性放到字典列表和类标签中
# #Python2.x
# #allElectronicsData = open(r'AllElectronics.csv', 'rb')
# #reader = csv.reader(allElectronicsData)
# #headers = reader.next()
# #上面的语句在python3.X会报错，'_csv.reader' object has no attribute 'next'
# #在python3.x需改为如下语句
# allElectronicsData = open(r'test.csv', 'rt')
# # allElectronicsData = open(r'test.csv', encoding = "gb18030", errors = "ignore")
#
# reader = csv.reader(allElectronicsData)
# headers = next(reader)
#
# print(headers)
# #['RID', 'age', 'income', 'student', 'credit_rating', 'class_buys_computer']
#
# featureList = []
# labelList = []
#
# for row in reader:
#     labelList.append(row[len(row)-1])
#     rowDict = {}
#     for i in range(1, len(row)-1):
#         rowDict[headers[i]] = row[i]
#     featureList.append(rowDict)
#
# print(featureList)
# '''
# [{'age': 'youth', 'credit_rating': 'fair', 'income': 'high', 'student': 'no'},
# {'age': 'youth', 'credit_rating': 'excellent', 'income': 'high', 'student': 'no'},
# {'age': 'middle_aged', 'credit_rating': 'fair', 'income': 'high', 'student': 'no'},
# {'age': 'senior', 'credit_rating': 'fair', 'income': 'medium', 'student': 'no'},
# {'age': 'senior', 'credit_rating': 'fair', 'income': 'low', 'student': 'yes'},
# {'age': 'senior', 'credit_rating': 'excellent', 'income': 'low', 'student': 'yes'},
# {'age': 'middle_aged', 'credit_rating': 'excellent', 'income': 'low', 'student': 'yes'},
# {'age': 'youth', 'credit_rating': 'fair', 'income': 'medium', 'student': 'no'},
# {'age': 'youth', 'credit_rating': 'fair', 'income': 'low', 'student': 'yes'},
# {'age': 'senior', 'credit_rating': 'fair', 'income': 'medium', 'student': 'yes'},
# {'age': 'youth', 'credit_rating': 'excellent', 'income': 'medium', 'student': 'yes'},
# {'age': 'middle_aged', 'credit_rating': 'excellent', 'income': 'medium', 'student': 'no'},
# {'age': 'middle_aged', 'credit_rating': 'fair', 'income': 'high', 'student': 'yes'},
# {'age': 'senior', 'credit_rating': 'excellent', 'income': 'medium', 'student': 'no'}]
# '''
# #从表中可以看出是用字典储存，所以是无序的。
#
# # Vetorize features
# vec = DictVectorizer()
# dummyX = vec.fit_transform(featureList) .toarray()
#
# print("dummyX: " + str(dummyX))
# #将每一行转化为如下格式
# #youth  middle_age senor   high medium low   yes no   fair excellent    buy
# # 1        0         0      1     0     0     0   1    1     0           0
# '''
# dummyX:
# [[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
#  [ 0.  0.  1.  1.  0.  1.  0.  0.  1.  0.]
#  [ 1.  0.  0.  0.  1.  1.  0.  0.  1.  0.]
#  [ 0.  1.  0.  0.  1.  0.  0.  1.  1.  0.]
#  [ 0.  1.  0.  0.  1.  0.  1.  0.  0.  1.]
#  [ 0.  1.  0.  1.  0.  0.  1.  0.  0.  1.]
#  [ 1.  0.  0.  1.  0.  0.  1.  0.  0.  1.]
#  [ 0.  0.  1.  0.  1.  0.  0.  1.  1.  0.]
#  [ 0.  0.  1.  0.  1.  0.  1.  0.  0.  1.]
#  [ 0.  1.  0.  0.  1.  0.  0.  1.  0.  1.]
#  [ 0.  0.  1.  1.  0.  0.  0.  1.  0.  1.]
#  [ 1.  0.  0.  1.  0.  0.  0.  1.  1.  0.]
#  [ 1.  0.  0.  0.  1.  1.  0.  0.  0.  1.]
#  [ 0.  1.  0.  1.  0.  0.  0.  1.  1.  0.]]
# '''
# print(vec.get_feature_names())
# '''
# ['age=middle_aged', 'age=senior', 'age=youth',
#  'credit_rating=excellent', 'credit_rating=fair',
#  'student=no', 'student=yes']
# '''
# print("labelList: " + str(labelList))
# #labelList:
# #['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
#
# # vectorize class labels
# lb = preprocessing.LabelBinarizer()
# dummyY = lb.fit_transform(labelList)
# print("dummyY: " + str(dummyY))
# '''
# dummyY:
# [[0]
#  [0]
#  [1]
#  [1]
#  [1]
#  [0]
#  [1]
#  [0]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [0]]
# '''
#
# # Using decision tree for classification
# # clf = tree.DecisionTreeClassifier()
# '''
# clf就是生成的决策树，参数可以选择决策树的算法种类，这里使用entropy即ID3信息熵算法。
# '''
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(dummyX, dummyY)
# print("clf: " + str(clf))
#
#
# # Visualize model
# '''
# 创建.dot文件用于存放可视化决策树数据，决策树已经数值化，如果要还原属性到决策树，需要传入属性参数feature_names=vec.get_feature_names()
# '''
# with open("allElectronicInformationGainOri.dot", 'w') as f:
#     f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)
#
# '''
# 最后把生成的.dot文件转换成可视化的pdf文件，dot -Tpdf input.dot -o output.pdf
#
# '''
#
# #决策树生成后，用demo实例预测结果
#
# #取第一行数据，并稍做改动
# oneRowX = dummyX[0, :]
# print("oneRowX: " + str(oneRowX))
# #oneRowX: [ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
#
# newRowX = oneRowX
# newRowX[0] = 1
# newRowX[2] = 0
# print("newRowX: " + str(newRowX))
# #newRowX: [ 1.  0.  0.  0.  1.  1.  0.  0.  1.  0.]
# #predictedY = clf.predict(newRowX)
# '''
# 直接运行会报如下错误
#     "if it contains a single sample.".format(array))
# ValueError: Expected 2D array, got 1D array instead:
# array=[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.].
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# 提示需要reshape，所以入参改为newRowX.reshape(1,-1)
# reshape作用可参考http://www.cnblogs.com/iamxyq/p/6683147.html
# '''
# predictedY = clf.predict(newRowX.reshape(1,-1))
# print("predictedY: " + str(predictedY))
# #predictedY: [1]



















from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
# from sklearn.externals.six import StringIO

#sklearn对数据有格式要求，首先要对数据进行格式预处理。
# Read in the csv file and put features into list of dict and list of class label
#读取csv文件，并把属性放到字典列表和类标签中
#Python2.x
#allElectronicsData = open(r'AllElectronics.csv', 'rb')
#reader = csv.reader(allElectronicsData)
#headers = reader.next()
#上面的语句在python3.X会报错，'_csv.reader' object has no attribute 'next'
#在python3.x需改为如下语句
allElectronicsData = open(r'test1.csv', 'rt')
# allElectronicsData = open(r'test.csv', encoding = "gb18030", errors = "ignore")

reader = csv.reader(allElectronicsData)
headers = next(reader)

print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

#从表中可以看出是用字典储存，所以是无序的。

# Vetorize features
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

print("dummyX: " + str(dummyX))
#将每一行转化为如下格式
#youth  middle_age senor   high medium low   yes no   fair excellent    buy
# 1        0         0      1     0     0     0   1    1     0           0
'''
dummyX: 
[[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
 [ 0.  0.  1.  1.  0.  1.  0.  0.  1.  0.]
 [ 1.  0.  0.  0.  1.  1.  0.  0.  1.  0.]
 [ 0.  1.  0.  0.  1.  0.  0.  1.  1.  0.]
 [ 0.  1.  0.  0.  1.  0.  1.  0.  0.  1.]
 [ 0.  1.  0.  1.  0.  0.  1.  0.  0.  1.]
 [ 1.  0.  0.  1.  0.  0.  1.  0.  0.  1.]
 [ 0.  0.  1.  0.  1.  0.  0.  1.  1.  0.]
 [ 0.  0.  1.  0.  1.  0.  1.  0.  0.  1.]
 [ 0.  1.  0.  0.  1.  0.  0.  1.  0.  1.]
 [ 0.  0.  1.  1.  0.  0.  0.  1.  0.  1.]
 [ 1.  0.  0.  1.  0.  0.  0.  1.  1.  0.]
 [ 1.  0.  0.  0.  1.  1.  0.  0.  0.  1.]
 [ 0.  1.  0.  1.  0.  0.  0.  1.  1.  0.]]
'''
print(vec.get_feature_names())
'''
['age=middle_aged', 'age=senior', 'age=youth', 
 'credit_rating=excellent', 'credit_rating=fair', 
 'student=no', 'student=yes']
'''
print("labelList: " + str(labelList))
#labelList:
#['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY: " + str(dummyY))
'''
dummyY: 
[[0]
 [0]
 [1]
 [1]
 [1]
 [0]
 [1]
 [0]
 [1]
 [1]
 [1]
 [1]
 [1]
 [0]]
'''

# Using decision tree for classification
# clf = tree.DecisionTreeClassifier()
'''
clf就是生成的决策树，参数可以选择决策树的算法种类，这里使用entropy即ID3信息熵算法。
'''
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))


# Visualize model
'''
创建.dot文件用于存放可视化决策树数据，决策树已经数值化，如果要还原属性到决策树，需要传入属性参数feature_names=vec.get_feature_names()
'''
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

'''
最后把生成的.dot文件转换成可视化的pdf文件，dot -Tpdf input.dot -o output.pdf

'''

FP_rate, TP_rate, thresholds = roc_curve(dummyY, clf.predict(dummyX))
print(auc(FP_rate, TP_rate))
# 0.5
print(FP_rate)
print(TP_rate)
print(roc_auc_score(dummyY, clf.predict(dummyX)))
# 0.5


#决策树生成后，用demo实例预测结果
#
# #取第一行数据，并稍做改动
# oneRowX = dummyX[0, :]
# print("oneRowX: " + str(oneRowX))
# #oneRowX: [ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
#
# newRowX = oneRowX
# newRowX[0] = 1
# newRowX[2] = 0
# print("newRowX: " + str(newRowX))
# #newRowX: [ 1.  0.  0.  0.  1.  1.  0.  0.  1.  0.]
# #predictedY = clf.predict(newRowX)
# '''
# 直接运行会报如下错误
#     "if it contains a single sample.".format(array))
# ValueError: Expected 2D array, got 1D array instead:
# array=[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.].
# Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
# 提示需要reshape，所以入参改为newRowX.reshape(1,-1)
# reshape作用可参考http://www.cnblogs.com/iamxyq/p/6683147.html
# '''
# predictedY = clf.predict(newRowX.reshape(1,-1))
# print("predictedY: " + str(predictedY))
# #predictedY: [1]