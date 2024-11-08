import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
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
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 将标签从字符串转换为数值
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 初始化模型
models = {
    "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "支持向量机": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "朴素贝叶斯": MultinomialNB()
}

# 训练模型并获取概率预测
prob_predictions = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train_encoded)  # 使用编码后的标签训练
    prob_predictions[name] = model.predict_proba(X_test_tfidf)

# 使用加权平均的方式进行模型融合
weights = {"逻辑回归": 0.3, "随机森林": 0.1, "支持向量机": 0.3, "XGBoost": 0.1, "朴素贝叶斯": 0.2}
weighted_probs = np.zeros_like(list(prob_predictions.values())[0])
for name, probs in prob_predictions.items():
    weighted_probs += weights[name] * probs

# 获得加权融合后的预测索引
weighted_predictions = np.argmax(weighted_probs, axis=1)

# 将数值预测索引转换回标签
unique_labels = models["逻辑回归"].classes_  # 确保这里获取的是模型预期的类别标签数组
weighted_predictions_labels = [unique_labels[pred] for pred in weighted_predictions]

# 评估融合模型的准确率
accuracy_weighted_ensemble = accuracy_score(y_test, weighted_predictions_labels)

print(f"加权融合模型准确率: {accuracy_weighted_ensemble}")
