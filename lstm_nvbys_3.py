import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 步骤 1: 读取数据
file_path = 'preprocessed_douban_reviews.csv'
data = pd.read_csv(file_path)

# 将评分转换为情感标签：0 - 负面，1 - 中性，2 - 正面
data['sentiment'] = pd.cut(data['rating'], bins=[0, 2, 3, 5], right=True, labels=[0, 1, 2]).astype(int)
X = data['comment']
y = data['sentiment']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤 2: 文本向量化
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = max(max(len(x) for x in X_train_seq), max(len(x) for x in X_test_seq))
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length)

# 步骤 3: 定义和训练LSTM模型
input_text = Input(shape=(None,))
x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100)(input_text)
x = LSTM(128)(x)
features = Dense(128, activation='relu', name='features')(x)
predictions = Dense(3, activation='softmax')(features)  # 修改为3个输出节点

model = Model(inputs=input_text, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 添加早停以避免过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model.fit(X_train_pad, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# 步骤 4: 特征提取
feature_model = Model(inputs=model.input, outputs=model.get_layer('features').output)
X_train_features = feature_model.predict(X_train_pad)
X_test_features = feature_model.predict(X_test_pad)

# 步骤 5: 训练朴素贝叶斯分类器
# 因为朴素贝叶斯是基于概率的模型，理论上可以处理多分类问题，但我们需要正确地准备数据和调整代码
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_features, y_train)

# 使用朴素贝叶斯分类器对测试集进行预测
y_pred = nb_classifier.predict(X_test_features)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, accuracy_score

# 计算准确率
nb_accuracy = accuracy_score(y_test, y_pred)
print(f'Naive Bayes Accuracy: {nb_accuracy}')

# 使用朴素贝叶斯分类器对测试集进行预测
y_pred = nb_classifier.predict(X_test_features)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", conf_matrix)

# 计算精确度、召回率和F1分数
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("精确度:", precision)
print("召回率:", recall)
print("F1分数:", f1)

# 打印分类报告以获取每个类别的详细指标
print("分类报告:\n", classification_report(y_test, y_pred))
