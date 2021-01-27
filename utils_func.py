# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:09:05 2020

@author: Ema
"""

import pandas as pd
import random
        
def close2ret(close):
    
    logret = lambda x_0, x_1: pd.np.log(x_0/x_1) 
    n = close.shape[0]
    out = pd.np.zeros(n)
    for i in range(n):
        if i > 0:
            out[i] = logret(close[i], close[i-1])  
    return out

def row_by_n(array, len_row):
    
    n_array = array.shape[0]
    out = pd.np.zeros((n_array, len_row))
    for i in range(n_array):
        for j in range(len_row):
            if i > j:
                out[i, j] = array[i-j-1]
    return out

def discretize_array(array, ths):
    
    n = array.shape[0]
    out = pd.np.zeros(n)
    for i in range(n):
        
        if array[i] > ths:
            out[i] = 1
        elif array[i] < -ths:
            out[i] = -1
    
    return out

def make_random_state():
    return int(random.random()*(2**(32-1)))

def ema_distance_feature(close, periods):
    
    from ta.utils import ema
    ema = ema(close, periods)
    df = pd.DataFrame({'close':close, 'ema':ema})
    diff = df['close'] / df['ema'] - 1
    
    return pd.np.array(diff.fillna(value=0))

def bollinger_distance_feature(close, period, n_dev):
    
    import ta.volatility as vol
    up_band = vol.bollinger_hband(close, period, n_dev).fillna(value=0)
    lo_band = vol.bollinger_lband(close, period, n_dev).fillna(value=0)
    df = pd.DataFrame({'close':close, 'up':up_band, 'lo':lo_band}) 
    up_diff = pd.np.array(df['up'] / df['close'] - 1 )
    lo_diff = pd.np.array(df['lo'] / df['close'] - 1 )
    
    return up_diff, lo_diff

def rsi_feature(close, period):
    
    import ta.momentum as mom
    rsi = mom.rsi(close, period).fillna(value=0) / 100
    
    return pd.np.array(rsi)

    
    
    
    
