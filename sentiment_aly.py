import pandas as pd

# 加载预处理后的文件
file_path = 'preprocessed_douban_reviews.csv'
df = pd.read_csv(file_path)

# 根据评分分类情感
# 假设4和5是积极的，3是中性词汇，1和2是消极的

df['sentiment'] = pd.cut(df.rating, bins=[0, 2, 3, 5], right=True, labels=['negative', 'neutral', 'positive'])

# 计算情感分布

sentiment_distribution = df['sentiment'].value_counts(normalize=True) * 100

print(sentiment_distribution)

# 画出饼图
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
sentiment_distribution.plot.pie(autopct='%1.1f%%', startangle=90, labels=sentiment_distribution.index)
ax.set_ylabel('')  # 删除y轴标签
plt.title('Sentiment Distribution')
plt.show()
