import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

file_path = 'preprocessed_douban_reviews.csv'
df = pd.read_csv(file_path)

# 根据评分定义情感
df['sentiment'] = pd.cut(df.rating, bins=[0, 2, 3, 5], right=True, labels=['negative', 'neutral', 'positive'])


# 预处理文本并提取每个情感类别最常见单词
def extract_common_words(df, sentiment, n=10):
    # 根据情绪过滤词汇
    comments = df[df['sentiment'] == sentiment]['comment'].tolist()

    # 初始化 CountVectorizer 将文本转换为标记计数矩阵
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(comments)

    # 计算每个词的数量
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    # 返回常见词
    return words_freq[:n]


# 提取每个情感分类的常见词 数量10
positive_words = extract_common_words(df, 'positive')
negative_words = extract_common_words(df, 'negative')
neutral_words = extract_common_words(df, 'neutral')

print('积极词汇：', positive_words)
print('中性词汇：', neutral_words)
print('负面词汇：', negative_words)

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
#
#
# # 提取每个情感分类的常见词 数量10
# positive_words = extract_common_words(df, 'positive', 30)
# negative_words = extract_common_words(df, 'negative', 30)
# neutral_words = extract_common_words(df, 'neutral', 30)
#
# # 生成词云
# def generate_wordcloud(words_freq, title):
#     wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='simhei.ttf').generate_from_frequencies(dict(words_freq))
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
#
#
# # Positive words
# generate_wordcloud(positive_words, 'Positive Sentiment')
#
# # Neutral words
# generate_wordcloud(neutral_words, 'Neutral Sentiment')
#
# # Negative words
# generate_wordcloud(negative_words, 'Negative Sentiment')
