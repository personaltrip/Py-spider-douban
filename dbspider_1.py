import requests
from bs4 import BeautifulSoup
import time
import csv


def scrape_douban_reviews(movie_id, page_limit):
    base_url = "https://movie.douban.com/subject/{}/comments".format(movie_id)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Cookie': 'bid=VG_umaKeRFE; viewed="25787839"; ll="118170"; dbcl2="272204662:kYdmBbmQtxw"; push_noty_num=0; push_doumail_num=0; ck=G2ZF; ap_v=0,6.0; frodotk_db="3ae9a4d30fec91ceb0bcee792fc5b56a"'
    }

    with open('douban_reviews.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['rating', 'comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for start in range(0, page_limit * 20, 20):
            params = {
                'start': start,
                'limit': 20,
                'status': 'P',
                'sort': 'new_score'
            }
            response = requests.get(base_url, headers=headers, params=params)
            if response.status_code != 200:
                print(response.status_code)
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            for comment in soup.find_all('div', class_='comment'):
                try:
                    rating = comment.find('span', class_=lambda x: x and 'rating' in x).get('class')[0]
                    comment_text = comment.find('span', class_='short').text
                    writer.writerow({'rating': rating, 'comment': comment_text})
                except AttributeError:
                    continue

            time.sleep(3)  # 避免给豆瓣服务器造成过大压力


if __name__ == '__main__':
    movie_id = '34841067'  # 《你好李焕英》

    page_limit = 40
    scrape_douban_reviews(movie_id, page_limit)