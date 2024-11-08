import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split

# 加载数据
file_path = 'preprocessed_douban_reviews.csv'  # 请根据实际情况修改路径
data = pd.read_csv(file_path)

# 分词的评论文本转换为列表
sentences = [comment.split() for comment in data['comment']]

# 训练word2vec模型
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)


# 向量化评论文本
def vectorize_comment(comment, model):
    vectors = [model.wv[word] for word in comment if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


comment_vectors = np.array([vectorize_comment(comment, model_w2v) for comment in sentences])

# 分割数据
X = np.expand_dims(comment_vectors, axis=1)
y = data['rating'].values  # 假设我们的任务是预测评分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建LSTM模型
model_lstm = Sequential()
model_lstm.add(SpatialDropout1D(0.2))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))  # 假设是二分类任务，对于回归任务可移除激活函数

model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model_lstm.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# 评估模型在测试集上的表现
test_loss, test_acc = model_lstm.evaluate(X_test, y_test, verbose=2)
print(f'Test Accuracy: {test_acc:.4f}')

# 检查训练结果
import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
