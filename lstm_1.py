import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam

# 加载数据
file_path = 'preprocessed_douban_reviews2.csv'
df = pd.read_csv(file_path)

# 假设评分4和5为正面情绪，1和2为负面情绪
df['sentiment_label'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)

# 文本预处理
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['comment'])
sequences = tokenizer.texts_to_sequences(df['comment'])
padded_sequences = pad_sequences(sequences, maxlen=200)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment_label'], test_size=0.2, random_state=42)

# 构建LSTM模型
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 模型训练
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# 模型评估
test_loss, test_acc = model.evaluate(X_test, y_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 进行预测
y_pred_probs = model.predict(X_test)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs.ravel()]

# 计算准确率，已经在模型评估时计算
print(f'Test Accuracy: {test_acc}')

# 计算精确度
precision = precision_score(y_test, y_pred)
print(f'Precision: {precision}')

# 计算召回率
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall}')

# 计算F1分数
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1}')


# # 检查训练结果
# import matplotlib.pyplot as plt
#
# # 绘制训练 & 验证的准确率值
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
