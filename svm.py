import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline

# 加载数据集
file_path = 'preprocessed_douban_reviews.csv'  # 请替换为实际文件路径
data = pd.read_csv(file_path)

# 将评分转换为三分类情感标签
# 正面: 4或5, 中性: 3, 负面: 1或2
data['sentiment_3class'] = data['rating'].apply(lambda x: 2 if x >= 4 else (1 if x == 3 else 0))

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data['comment'], data['sentiment_3class'], test_size=0.2, random_state=42)

# 创建管道，先将文本转换为TF-IDF向量，然后应用SVM
pipeline = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', decision_function_shape='ovo'))

# 训练模型
pipeline.fit(X_train, y_train)

# 在测试集上预测
y_pred = pipeline.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

