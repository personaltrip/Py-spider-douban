import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

file_path = 'preprocessed_douban_reviews.csv'
df = pd.read_csv(file_path)

# Step 2: 使用TF-IDF提取特征词汇
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df['comment'])

# Step 3: 定义情感标签
#df['sentiment_label'] = df['rating'].apply(lambda x: 'positive' if x > 3 else 'negative')
df['sentiment_label'] = df['rating'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))

# Step 4: 分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df['sentiment_label'], test_size=0.2, random_state=42)

# Step 5: 训练模型
nb_classifier = MultinomialNB(alpha=1)
nb_classifier.fit(X_train, y_train)

# Step 6: 模型评估
y_pred = nb_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 加权平均计算精确度、召回率和F1分数
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# 打印评估指标
print("准确率:", accuracy)
print("精确度（加权）:", precision_weighted)
print("召回率（加权）:", recall_weighted)
print("F1分数（加权）:", f1_weighted)
