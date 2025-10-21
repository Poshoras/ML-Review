<img width="572" height="351" alt="截屏2025-10-21 10 02 50" src="https://github.com/user-attachments/assets/88c9c381-3566-4cab-aff7-ab455b7b9025" />![image](https://github.com/user-attachments/assets/074352d5-620b-453a-96c6-490f871fb977)# SVM
### 核心思想
分类算法，核心参数是一个分割线，和两个超平面。

用点到线推出距离公式，作为目标函数（距离点越远越好），约束是所有点都在超平面之外。

但是有的点不听话，所以不得不用松弛变量网开一面。

但是在优化的时候，优化无法使用梯度下降。因为有一个max函数导致其不可导。（梯度下降就是通过不断沿着函数下降最快的方向（负梯度方向）移动参数，来逐步最小化损失函数。）

解决方法是用次梯度下降法，利用其凸函数的性质给出一个近似的导数，虽然不是最优但是也能用。

### 代码
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1️⃣ 加载示例数据集（这里用鸢尾花 iris）
iris = datasets.load_iris()
X = iris.data[:, :2]  # 只取前两个特征，方便可视化
y = iris.target

# 2️⃣ 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ 创建并训练 SVM 模型
model = SVC(kernel='linear')  # 线性核函数
model.fit(X_train, y_train)

# 4️⃣ 预测
y_pred = model.predict(X_test)

# 5️⃣ 计算准确率
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6️⃣ 输出支持向量信息
print("Number of support vectors:", len(model.support_))
```




# LDA和PCA
都是用来降维来解决维度超载的问题。

最大区别：LDA是监督学习，PCA是非监督学习。


# Bagging ； Boosting ； AdaBoosting
这是集成学习的一种思路，并不是一个独立的算法。

Bagging：n次独立抽样，然后投票或者均值得到结果，可以并行计算；

Boosting：n次迭代计算，每运行一次就顺着Loss function梯度下降的方向走一点，不能并行计算。相当于是学习了先前模型的经验，也算是Bagging的一个变种。

AdaBoost：Boosting思想的运用，每一次用基础模型训练以后，对那些被误分类的样本，增加权重，使其在下一次训练时被误分类的惩罚加大，以此矫正模型的性能。

XGBoost：实现原理复杂，但是可以并行计算，特别牛逼，设置参数也是多到离谱。挑几个比较特别的点说一下：1.损失函数采用泰勒展开式得到其一阶，两阶导数，梯度下降的更快更准。2.对树的棵数量和权重均采用正则化系数惩罚，防止其过拟合。

