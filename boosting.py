import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 加载数据集
data_path = 'preprocessed_douban_reviews.csv'
data = pd.read_csv(data_path)

# 将评分转换为情感类别
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

# 初始化LabelEncoder
label_encoder = LabelEncoder()

# 将文本标签转换为整数
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 创建XGBoost分类器实例
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# 训练模型，使用转换后的标签
xgb_model.fit(X_train_tfidf, y_train_encoded)

# 使用训练好的模型进行预测
predictions_encoded = xgb_model.predict(X_test_tfidf)

# 将预测的数值标签转换回文本标签
predictions = label_encoder.inverse_transform(predictions_encoded)

# 评估模型的准确率
accuracy = accuracy_score(y_test, predictions)

print(f"Boosting模型准确率: {accuracy}")
