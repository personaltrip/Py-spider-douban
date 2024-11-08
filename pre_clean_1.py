import re
import string
import pandas as pd
import jieba
# from zhon.hanzi import stops

def preprocess_reviews(csv_file):
    df = pd.read_csv(csv_file)

    # 去除评论两端的空白字符
    df['comment'] = df['comment'].str.strip()

    # 所有中英文标点符号
    punctuation_pattern = r"[{}]+".format(re.escape(string.punctuation + '，。？！；：、‘’“”《》~（）…【】「」'))
    # punctuation_pattern = r"[{}]+".format(re.escape(string.punctuation + stops))

    # 对 'comment' 列应用正则替换
    df['comment'] = df['comment'].str.replace(punctuation_pattern, '')

    # 读取停用词文件
    def read_stopwords(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f]
        return set(stopwords)

    # 读取停用词列表
    stopwords = read_stopwords('stopwd.txt')

    # # 载入情感词典
    # jieba.load_userdict("dic/BosonNLP/BosonNLP_sentiment_score.txt")
    # jieba.load_userdict("dic/台湾大学NTUSD简体中文情感词典/ntusd-negative.txt")
    # jieba.load_userdict("dic/台湾大学NTUSD简体中文情感词典/ntusd-positive.txt")

    # 启动paddle模式
    # jieba.enable_paddle()

    # 分词函数
    def tokenize_and_remove_stopwords(text):
        # words = jieba.cut(str(text), use_paddle=True)
        words = jieba.cut(str(text))
        words = [word.strip() for word in words if word.strip() and word.strip() not in stopwords]
        return ' '.join(words)

    # 对 'comment' 列应用分词和去除停用词的函数
    df['comment'] = df['comment'].apply(tokenize_and_remove_stopwords)

    # 将评分转换为数值
    df['rating'] = df['rating'].apply(
        lambda x: (int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0) * 0.1)

    # 保存预处理后的数据
    df.to_csv('preprocessed_douban_reviews.csv', index=False)


def clean_nan(csv_file):
    # Load the provided CSV file
    df = pd.read_csv(csv_file)

    # Drop rows with any column having NA/null data
    df_cleaned = df.dropna()

    # Save the cleaned DataFrame back to a new CSV file
    cleaned_file_path = 'preprocessed_douban_reviews.csv'
    df_cleaned.to_csv(cleaned_file_path, index=False)

if __name__ == '__main__':
    preprocess_reviews('douban_reviews.csv')
    clean_nan('preprocessed_douban_reviews.csv')
