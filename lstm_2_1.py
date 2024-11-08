import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping

# 加载数据
file_path = 'preprocessed_douban_reviews.csv'
df = pd.read_csv(file_path)

# 修改情感标签为三个类别
df['sentiment_label'] = pd.cut(df['rating'], bins=[0, 2, 3, 5], labels=[0, 1, 2], right=True)

# 文本预处理
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['comment'])
sequences = tokenizer.texts_to_sequences(df['comment'])
padded_sequences = pad_sequences(sequences, maxlen=200)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment_label'], test_size=0.2, random_state=42)

# 将标签转换为分类形式
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 构建LSTM模型
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')  # 修改为3个输出节点
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 添加早停以避免过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# 模型训练
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 模型评估
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# 可以使用sklearn的metrics库计算更多评估指标，但需要预测测试集
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=-1)
y_true = y_test.argmax(axis=-1)

# 计算精确度、召回率和F1分数
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
