# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:33:01 2019

@author: Ema
"""

import requests as rq
import time
from datetime import datetime

class PublicApi:
    def __init__(self):    
        self.url_base = 'https://www.deribit.com/api/v2/'
    
    def get_prices_by_time(self, instr, start, end, count):
        url = '{}public/get_last_trades_by_instrument_and_time?instrument_name={}&end_timestamp={}&count={}&include_old=true&start_timestamp={}'.format(self.url_base, instr, str(end), count, str(start))
        return self.process(url)
    
    def get_instruments(self, currency):
        url = '{}public/get_instruments?currency={}'.format(self.url_base, currency)
        return self.process(url)
    
    def process(self, url):
        try:
            return rq.get(url).json()['result']
        except Exception as e:
            print(e)
            print(datetime.now())
            print('Sleeping for 30 secs...')
            time.sleep(30)
            return self.process(url)