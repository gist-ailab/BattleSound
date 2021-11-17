## Crawling the link from the keywords

# Load the library
import os, sys
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re
import time
import pytube
import glob
import subprocess

import pprint
import pickle
import numpy as np

def Crawling_from_url(url, folder_dir, file_name):
    try:
        yt = pytube.YouTube(url)
        vid = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().last().download(folder_dir, file_name)
        return True
    except:
        print(url + ' : Not downloaded')
        return False

# # Crawling with chrome web driver
# options = webdriver.ChromeOptions()
# options.add_argument('headless')
# options.add_argument('window-size=1920x1080')
# options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
# options.add_argument("lang=ko_KR") # 한국어!
# driver = webdriver.Chrome('./chromedriver.exe', options = options)
#
# # Function for conditionning the time (sec)
# def Time_Condition(time_data, min_sec, max_sec):
#     if (':' not in time_data):
#         return False
#
#     time_value = list(map(int, time_data.split(':')))
#     time_sec = sum([val * (60 ** (len(time_value) - (1 + ix))) for ix, val in enumerate(time_value)])
#
#     if min_sec <= time_sec <= max_sec:
#         return True
#
#     else:
#         return False
#
# ## Crawling
# keywords=['브롤스타즈', '슈퍼마리오', '클래시로얄']
#
# for keyword in keywords:
#
#     play_time = []
#     site = []
#
#
#     game_name = keyword
#     # keyword = keyword + ' ' + '무편집 플레이'
#     print('Web-Crawling with %s keyword' %keyword)
#
#     url = 'https://www.youtube.com/results?search_query=' + keyword
#
#     driver.get(url)
#     time.sleep(2)
#     body = driver.find_element_by_tag_name("body")
#
#     num_of_pagedowns = 200
#     while num_of_pagedowns:
#         body.send_keys(Keys.PAGE_DOWN)
#         time.sleep(0.2)
#         num_of_pagedowns -= 1
#
#         if driver.find_elements_by_xpath("//*[contains(text(), '결과가 더 이상 없습니다.')]") != []:
#             break
#
#     html = driver.page_source
#     soup = BeautifulSoup(html, 'html.parser')
#
#     # a 태그 : 다른 페이지와의 연결자c
#     # href 속성 : 연결할 주소 지정
#
#     li = soup.find("div", {"id": "content"}).find_all("a", href=re.compile("^(/watch\?v\=)(.)*$"), id='thumbnail')
#
#     for link in soup.find("div", {"id": "content"}).find_all("a", href=re.compile("^(/watch\?v\=)(.)*$"), id='thumbnail'):
#         word = link.get_text().strip()
#         word = re.findall('\d+', word)
#         if len(word) == 1:
#             continue
#
#         word = ':'.join(word)
#
#         play_time.append(word)
#         site.append(link.attrs['href'])
#
#     time_label = [Time_Condition(imp_time, 120, 3600) for imp_time in play_time]
#     site = np.array(site)
#     selected = site[time_label]
#
#     print('Filtered : %d, Not filtered : %d' % (len(selected), len(site)))
#
#     with open(f'{game_name}_selected.pkl', 'wb') as f:
#         pickle.dump(selected, f)

# Preview the list
# pkl_list = glob('./pickle/*.pkl')
#
# for pkl in pkl_list:
#     game_name = pkl.split('/')[-1].split('_')[0]
#     print(game_name)
#     with open(pkl, 'rb') as f:
#         web_list= pickle.load(f)
#
#     title_list = []
#
#     if not os.path.isdir(f'/SSD1/datasets/Battle_1st/video/{game_name}'):
#         os.makedirs(f'/SSD1/datasets/Battle_1st/video/{game_name}')
#
#     folder_dir = f'/SSD1/datasets/Battle_1st/video/{game_name}'
#     for ix, link in enumerate(web_list):
#         try:
#             yt = pytube.YouTube(link)
#             # if ('무편집' not in yt.title) | (game_name not in yt.title):
#             if game_name not in yt.title:
#                 continue
#
#             vid = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().last().download(folder_dir)
#             print('# %d files are downloaded' % (ix + 1))
#
#         except:
#             print('# %d files are not loaded' % (ix + 1))
