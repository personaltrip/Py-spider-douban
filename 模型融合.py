import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import numpy as np

# 加载数据集
data_path = 'preprocessed_douban_reviews.csv'
data = pd.read_csv(data_path)

# 将评分转换为三个情感类别的函数
def categorize_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating <= 4:
        return 'neutral'
    else:
        return 'positive'

# 应用函数创建情感列
data['sentiment'] = data['rating'].apply(categorize_sentiment)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['comment'], data['sentiment'], test_size=0.2, random_state=42)

# 使用TF-IDF对文本数据进行向量化处理
tfidf = TfidfVectorizer(max_features=5000)

# 初始化模型
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, random_state=42)
mlp = MLPClassifier(random_state=42, max_iter=300)
nb = MultinomialNB()

# 为每个模型创建管道
pipeline_lr = make_pipeline(tfidf, lr)
pipeline_rf = make_pipeline(tfidf, rf)
pipeline_svm = make_pipeline(tfidf, svm)
pipeline_mlp = make_pipeline(tfidf, mlp)
pipeline_nb = make_pipeline(tfidf, nb)

# 训练每个模型
pipeline_lr.fit(X_train, y_train)
pipeline_rf.fit(X_train, y_train)
pipeline_svm.fit(X_train, y_train)
pipeline_mlp.fit(X_train, y_train)
pipeline_nb.fit(X_train, y_train)

# 获取每个模型的准确率
accuracy_lr = accuracy_score(y_test, pipeline_lr.predict(X_test))
accuracy_rf = accuracy_score(y_test, pipeline_rf.predict(X_test))
accuracy_svm = accuracy_score(y_test, pipeline_svm.predict(X_test))
accuracy_mlp = accuracy_score(y_test, pipeline_mlp.predict(X_test))
accuracy_nb = accuracy_score(y_test, pipeline_nb.predict(X_test))

# 获取每个模型的概率预测
prob_lr = pipeline_lr.predict_proba(X_test)
prob_rf = pipeline_rf.predict_proba(X_test)
prob_svm = pipeline_svm.predict_proba(X_test)
prob_mlp = pipeline_mlp.predict_proba(X_test)
prob_nb = pipeline_nb.predict_proba(X_test)

# 计算所有模型预测的平均概率
avg_probs = (prob_lr + prob_rf + prob_svm + prob_mlp + prob_nb) / 5
predictions_ensemble = np.argmax(avg_probs, axis=1)

# 将数值预测转换回标签
unique_labels = pipeline_lr.classes_
predictions_ensemble_labels = [unique_labels[pred] for pred in predictions_ensemble]

# 评估融合模型的准确率
accuracy_ensemble = accuracy_score(y_test, predictions_ensemble_labels)

# 输出每个模型及融合模型的准确率
print("逻辑回归模型准确率:", accuracy_lr)
print("随机森林模型准确率:", accuracy_rf)
print("支持向量机模型准确率:", accuracy_svm)
print("多层感知器模型准确率:", accuracy_mlp)
print("朴素贝叶斯模型准确率:", accuracy_nb)
print("融合模型准确率:", accuracy_ensemble)
