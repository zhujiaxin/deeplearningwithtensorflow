# -*- coding:utf-8 -*-
import requests
import os
import time


def download(url, path):
    if os.path.exists(path):
        os.remove(path)
    start = time.time()
    size = 0
    respone = requests.get(url, stream=True)
    chunk_size = 1024
    content_size = int(respone.headers['Content-Length'])
    if respone.status_code == 200:
        print('all size is {:2f} MB'.format(content_size / chunk_size / 1024))
        with open(path, 'wb') as file:
            for data in respone.iter_content(chunk_size=chunk_size):
                file.write(data)
                size += len(data)
                print('\r' + 'downloading file:' +
                      '>'*(size*50 // content_size) +
                      '{:.2f}% '.format(size*100 / content_size), end='')
        end = time.time()
        print('\n download use {:.2f} ms'.format(end-start))
