import requests
from bs4 import BeautifulSoup
import csv


def fetch_douban_top250():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    }
    movie_ids = []

    for start in range(0, 250, 25):
        url = f'https://movie.douban.com/top250?start={start}'
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        for item in soup.find_all('div', class_='item'):
            movie_url = item.find('a')['href']
            movie_id = movie_url.split('/')[-2]
            movie_ids.append(movie_id)

    return movie_ids


def save_to_csv(movie_ids):
    with open('douban_top250_movie_ids.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Movie ID'])
        for movie_id in movie_ids:
            writer.writerow([movie_id])


if __name__ == '__main__':
    movie_ids = fetch_douban_top250()
    save_to_csv(movie_ids)
    print('豆瓣Top 250电影ID已保存到CSV文件。')
