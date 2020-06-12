
# ===========
# 一、特征工程
# ===========

# 1.标准化

from sklearn.preprocessing import StandardScaler
data = StandardScaler().fit_transform(data)

# 2.区间缩放

from sklearn.preprocessing import MinMaxScaler
data = MinMaxScaler().fit_transform(data)
# 3.归一化，便于计算梯度下降
from sklearn.preprocessing import Normalizer
data = Normalizer().fit_transform(data)
# 4.定量特征二值化（大于epsiloin的为1， 小于epsiloin的为0）
from sklearn.preprocessing import Binarizer
data = Binarizer(threshold=epsilon).fit_transform(data)
# 5.类别特征转换成数值特征
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
data = vec.fit_transform(data.to_dict(orient=“recoed”))
# 6.卡方检验，选择最好的特征
from sklearn.feature_extraction import SelectKBest
from sklearn.feature_extraction import chi2
skb = SelectKBest(chi2, k = 10).fit（X,Y）
x_train = skb.transform（x_train）
# 7.互信息法
from sklearn.feature_selection import SelectKBest
from minepy import iNE
def mic(x, y):
m = MINE（）
m.compute_score(x, y)
return (m.mic(), 0.5)

# 选择K个最好的特征，返回特征选择后的数据

data = SelectKBest(lambda X, Y：array(map(lambd x:mic(x, y), X.T, k=2).fit_transform(x, y)))

# 8.主成分分析

from sklearn.decomposition import PCA
estimator = PCA(n_components=2)
x_pca = estimator.fit_transform(x_data)

# ===========
# 二、学习方法
# ===========
# 1.划分数据集和测试集

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# 2.交叉验证集
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, Y, cv=10)

# 3.训练

from sklearn import learnAlgorithm #对应的学习算法名称
la = learnAlgorithm()
la.fit(X_train, Y_train)
score = la.score(X_test, Y_test)
# 4.随机梯度下降
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X, y)
from sklearn.linear_model import SGDRegressor
rlf = SGDRegressor()
rlf.fit(X, y)
# 5.支持向量机，分类和回归
from sklearn.svm import SVC
svc_linear = SVC(kernel=“linear”)#选择不同的核函数
from sklearn.svm import SVR
svm_linear = SVR(kernel=“linear”)
# 6.朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB（）
# 7.决策树
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion=“entropy”,max_depth=4,min_samples_leaf=5)#指定最大深度和最小的样本数，防止过拟合
# 8.随机森林
from sklearn.ensemble import RandonForestClassifier
rfc = RandonForestClassifier(max_depth=3,min_samples_leaf=5)
# 9.梯度提升树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(max_depth=3, min_samples_leaf=5)
# 10.极限回归森林
from sklearn.ensemble import ExtraTreesRegressor

# ===========
# 三、模型评估
# ===========
# 1.获取精确率，召回率等等

from sklearn import metrics
accuracy_rate = metrics.accuracy_score(y_test, y_predict)
metrics.classification_report(y_test, y_predict, target_names = data.target_names)#可以获取准确率,召回率等数据

# 2.交叉验证

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, Y, cv=10)

# X,Y为ndarray类型，csv读取进来的数据可以用df.values来转 # Pipeline机制 # pipeline机制实现了对全部步骤的流式化封装和管理,应用于参数 # 集在数据集上的重复使用.Pipeline对象接受二元tuple构成的 # list,第一个元素为自定义名称,第二个元素为sklearn中的 # transformer或estimator,即处理特征和用于学习的方法. # 以朴素贝叶斯为例,根据处理特征的不同方法有以下代码

clf_1 = Pipeline([(‘count_vec’, CountVectorizer()), (‘mnb’, MultinomialNB())])
clf_2 = Pipeline([(‘hash_vec’, HashingVectorizer(non_negative=True)), (‘mnb’, MultinomialNB())])
clf_3 = Pipeline([(‘tfidf_vec’, TfidfVectorizer()), (‘mnb’, MultinomialNB())])

# ===========
# 四、特征选择
# ===========
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(feature_selection.chi2,percentile=per)
X_train = fs.fit_transform(Xtrain, Y_train)
