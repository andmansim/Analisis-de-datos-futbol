from bs4 import BeautifulSoup
import requests

url = 'https://www.uefa.com/uefachampionsleague/statistics/players/?sortBy=minutes_played_official&order=aschttps://www.uefa.com/uefachampionsleague/statistics/players/?sortBy=minutes_played_official&order=asc'
response = requests.get(url)
html = response.text
soup = BeautifulSoup(html, 'html.parser')
players = soup.find_all('td', class_='player')
