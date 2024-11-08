import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

file_path = 'preprocessed_douban_reviews.csv'
df = pd.read_csv(file_path)

# 初始化并使用Tfidf进行特征提取
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # 简单起见设置为1000个特征
tfidf_matrix = tfidf_vectorizer.fit_transform(df['comment'])

# 提取特征名和对应的tfidf分数
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.sum(axis=0)
tfidf_scores_array = tfidf_scores.A1  # 将矩阵转化为平面数组

# 为tfidf分数创建dataframe
tfidf_df = pd.DataFrame({'feature': feature_names, 'tfidf': tfidf_scores_array})

# 对dataframe进行tfidf分数的降序排序来显示重要的词汇
tfidf_sorted_df = tfidf_df.sort_values(by='tfidf', ascending=False)

# 显示十个tfidf分数最高的词汇
print(tfidf_sorted_df.head(10))

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
#
#
# # 生成中文词云的函数，这里使用TF-IDF分数最高的词汇
# def generate_wordcloud_chinese(tfidf_sorted_df):
#     # 提取词汇和对应的TF-IDF分数
#     words = {row['feature']: row['tfidf'] for index, row in tfidf_sorted_df.iterrows()}
#     wordcloud = WordCloud(width=800, height=400, background_color='white',
#                           font_path='simhei.ttf').generate_from_frequencies(words)
#
#     # 显示词云
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.show()
#
#
# # 使用TF-IDF分数最高的10个词汇生成词云
# generate_wordcloud_chinese(tfidf_sorted_df.head(30))

