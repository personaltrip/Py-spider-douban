# 导入必要的库
import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Embedding, LSTM, Dense, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping


# 加载数据
df = pd.read_csv('preprocessed_douban_reviews.csv')

# 修改情感标签为三个类别
df['sentiment_label'] = pd.cut(df['rating'], bins=[0, 2, 3, 5], labels=[0, 1, 2], right=True)

# 文本预处理
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['comment'])
sequences = tokenizer.texts_to_sequences(df['comment'])
padded_sequences = pad_sequences(sequences, maxlen=200)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment_label'], test_size=0.2, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 构建LSTM模型
lstm_model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])

# 编译模型
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 添加早停以避免过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# 训练LSTM模型
lstm_model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# 修改模型以提取特征
feature_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.layers[-2].output)
X_train_features = feature_extractor.predict(X_train)
X_test_features = feature_extractor.predict(X_test)

# 将y_train和y_test转换回原来的标签形式以用于SVM
y_train_svm = [np.argmax(y) for y in y_train]
y_test_svm = [np.argmax(y) for y in y_test]

# 训练SVM模型
svm_model = SVC(C=1.0, kernel='linear', probability=True)
svm_model.fit(X_train_features, y_train_svm)

# 预测测试集
y_pred = svm_model.predict(X_test_features)

# 评估SVM模型
print(classification_report(y_test_svm, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 预测测试集的标签
y_pred = svm_model.predict(X_test_features)

# 因为SVM模型使用的是非one-hot编码的标签，所以需要确保y_test也是非one-hot编码的标签
y_test_non_one_hot = [np.argmax(y) for y in y_test]

# 计算各种评估指标
accuracy = accuracy_score(y_test_non_one_hot, y_pred)
precision = precision_score(y_test_non_one_hot, y_pred, average='weighted')
recall = recall_score(y_test_non_one_hot, y_pred, average='weighted')
f1 = f1_score(y_test_non_one_hot, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
