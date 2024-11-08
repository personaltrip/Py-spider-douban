import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.utils import to_categorical

# 载入数据
file_path = 'preprocessed_douban_reviews.csv'
data = pd.read_csv(file_path)

# 根据评分设置情感标签
# 1-2: Negative (0), 3: Neutral (1), 4-5: Positive (2)
data['sentiment'] = pd.cut(data['rating'], bins=[0, 2, 3, 5], labels=[0, 1, 2], right=True)

# 准备分词器
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['comment'])
sequences = tokenizer.texts_to_sequences(data['comment'])

# 填充序列长度相等
max_length = max(len(x) for x in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['sentiment'], test_size=0.2, random_state=42)

# 将标签转换为分类
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# 创建CNN模型
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=max_length),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(units=24, activation='relu'),
    Dense(units=3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)

from keras.metrics import  Precision, Recall
import numpy as np

# 计算模型在测试集上的准确度
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# 预测测试集
y_pred = model.predict(X_test)

# 将预测结果和真实标签从独热编码转换为类别编号
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 计算精确度、召回率和F1分数
precision = Precision()
recall = Recall()
precision.update_state(y_true_classes, y_pred_classes)
recall.update_state(y_true_classes, y_pred_classes)
f1_score = 2 * (precision.result().numpy() * recall.result().numpy()) / (precision.result().numpy() + recall.result().numpy() + 1e-7)

print(f'Precision: {precision.result().numpy():.4f}')
print(f'Recall: {recall.result().numpy():.4f}')
print(f'F1 Score: {f1_score:.4f}')
