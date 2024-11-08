import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# 加载数据
file_path = 'preprocessed_douban_reviews.csv'
df = pd.read_csv(file_path)

# 根据评分调整情感标签
df['sentiment_label'] = pd.cut(df['rating'], bins=[0, 2, 3, 5], right=True, labels=[0, 1, 2])

# 文本预处理
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['comment'])
sequences = tokenizer.texts_to_sequences(df['comment'])
padded_sequences = pad_sequences(sequences, maxlen=200)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment_label'], test_size=0.2, random_state=42)

# 将标签转换为分类格式（one-hot encoding）
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# 构建LSTM模型
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')  # 输出层改为3个神经元，使用softmax激活函数
])

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# 模型训练
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test))

# 模型评估
test_loss, test_acc = model.evaluate(X_test, y_test)

print(test_acc)