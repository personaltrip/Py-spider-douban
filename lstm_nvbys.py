import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 步骤 1: 读取数据
file_path = 'preprocessed_douban_reviews2.csv'
data = pd.read_csv(file_path)

# 将评分转换为情感标签
data['sentiment'] = (data['rating'] > 3).astype(int)
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
predictions = Dense(1, activation='sigmoid')(features)

model = Model(inputs=input_text, outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_pad, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=3)])

# 步骤 4: 特征提取
feature_model = Model(inputs=model.input, outputs=model.get_layer('features').output)
X_train_features = feature_model.predict(X_train_pad)
X_test_features = feature_model.predict(X_test_pad)

# 步骤 5: 训练朴素贝叶斯分类器
nb_classifier = GaussianNB()
nb_classifier.fit(X_train_features, y_train)
nb_score = nb_classifier.score(X_test_features, y_test)

print("朴素贝叶斯分类器的准确率:", nb_score)

