## Crawling the link from the keywords

# Load the library
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


# Crawling with chrome web driver
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36")
options.add_argument("lang=ko_KR") # 한국어!
driver = webdriver.Chrome('../chromedriver.exe', options = options)

# Preview the list
with open('link_selected.pkl', 'rb') as f:
    web_list= pickle.load(f)

title_list = []

for ix, link in enumerate(web_list):
    try:
        yt = pytube.YouTube(link)
        vid = yt.streams.filter(only_audio=True, file_extension='mp4').all()

        title_list.append(yt.title)

        folder = 'F:/mp4_file'
        vid[0].download(folder)
        print('%d / %d files are downloaded' % (ix + 1, len(web_list)))

    except:
        print('%d / %d files are not loaded' % (ix + 1, len(web_list)))

# Save the titles of the downloaded links
with open('../title_list.pkl', 'wb') as f:
    pickle.dump(title_list, f)

