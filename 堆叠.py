import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

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
    ("逻辑回归", LogisticRegression(solver='saga', max_iter=5000, random_state=42)),
    ("支持向量机", SVC(probability=True, random_state=42)),
    ("朴素贝叶斯", MultinomialNB())
]

# 更改元模型为支持向量机
meta_model = SVC(probability=True, random_state=42)

# 重新构建堆叠分类器
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# 训练堆叠模型
stacked_model.fit(X_train_tfidf, y_train)

# 进行预测
y_pred = stacked_model.predict(X_test_tfidf)

# 评估融合模型的准确率
accuracy_stacked = accuracy_score(y_test, y_pred)

print(f"修改后的模型堆叠准确率: {accuracy_stacked}")