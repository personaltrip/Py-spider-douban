import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

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

# 定义基模型
base_models = [
    ("逻辑回归", LogisticRegression(max_iter=1000, random_state=42)),
    ("随机森林", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("支持向量机", SVC(probability=True, random_state=42)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ("朴素贝叶斯", MultinomialNB())
]

# 定义元模型
# 元模型通常是一个简单的模型，例如逻辑回归
meta_model = SVC(probability=True, random_state=42)

# 创建堆叠分类器
stacking_classifier = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# 训练堆叠模型
stacking_classifier.fit(X_train_tfidf, y_train)

# 预测
predictions = stacking_classifier.predict(X_test_tfidf)

# 评估堆叠模型的准确率
accuracy_stacked_ensemble = accuracy_score(y_test, predictions)

print(f"堆叠模型准确率: {accuracy_stacked_ensemble}")
