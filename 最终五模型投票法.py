import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

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

# 将标签字符串转换为数值
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 定义模型列表
models = [
    ("逻辑回归", LogisticRegression(max_iter=1000, random_state=42)),
    ("随机森林", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("支持向量机", SVC(probability=True, random_state=42)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ("朴素贝叶斯", MultinomialNB())
]

# 评估每个模型的准确率
for name, model in models:
    model.fit(X_train_tfidf, y_train_encoded)  # 使用转换后的标签进行训练
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test_encoded, predictions)  # 使用转换后的标签进行评估
    print(f"{name} 准确率: {accuracy}")

# 使用投票法创建集成模型
voting_classifier = VotingClassifier(estimators=models, voting='soft')

# 训练集成模型
voting_classifier.fit(X_train_tfidf, y_train_encoded)  # 使用转换后的标签进行训练

# 使用集成模型进行预测
ensemble_predictions = voting_classifier.predict(X_test_tfidf)

# 评估集成模型的准确率
accuracy_ensemble = accuracy_score(y_test_encoded, ensemble_predictions)  # 使用转换后的标签进行评估

print(f"投票法集成模型准确率: {accuracy_ensemble}")
