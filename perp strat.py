# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:54:39 2019

@author: Ema
"""

from deribit_public_api import PublicApi
from datetime import datetime
import matplotlib.pyplot as plt

from v_1 import ANNSignal

import time
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=Warning)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



start_exec = datetime.now()

class Portfolio:
    
    def __init__(self, data):
        
        self.data = data
    
        self.equity          = 1
        self.delta           = 2/100
        self.costs           = 0.00075
        self.avg_ba          = 2
        self.stop_loss_pct   = 0.02
        self.take_profit_pct = 0.03
        self.trail_profit    = (True, 0.01)
        
        self.pnl    = 0
        self.open_price = 0
        self.trade_journal = [] 
        self.use_kelly_delta = False
        self.stop_loss = self.stop_loss_pct
        self.take_profit = self.take_profit_pct
        
    def get_qty(self, price):
        if self.use_kelly_delta:
            
            kelly = self.kelly_delta()
            if kelly > 0.2:
                kelly = 0.2
            
            return pd.np.round(self.equity * kelly * price)
        
        else:
            
            return pd.np.round(self.equity * self.delta * price)
    
    def get_costs(self, price, qty):
        return self.costs*((qty * 10 / price))
    
    def get_pnl(self, enter, close, qty, direction):
        
        if direction == 'long':
            return (qty * 10) / enter - (qty * 10) / close
        
        if direction == 'short':
            return (qty * 10) / close - (qty * 10) / enter
        
    def kelly_delta(self):
        
        ev = lambda win_rate, win_ratio: win_ratio * win_rate - win_rate
        
        n = len(self.trade_journal)
        
        if self.trade_journal:
            
            if len(n) > 20:
                
                win_rate  = sum([1 for i in range(n) if self.trade_journal[i][8] > 0]) / n
                avg_win   = sum([i for i in range(n) if self.trade_journal[i][8] > 0]) / n
                avg_loss  = sum([i for i in range(n) if self.trade_journal[i][8] < 0]) / n
                win_ratio = avg_win / avg_loss
                
                if ev(win_rate, win_ratio) > 0:
                    kelly = win_rate - ((1 - win_rate) / win_ratio)
                    
                    self.take_profit = self.take_profit_pct * kelly / self.delta
                    self.stop_loss   = self.stop_loss_pct * kelly / self.delta
                    
                    return kelly
        
        self.take_profit = self.take_profit_pct
        self.stop_loss   = self.stop_loss_pct
        return self.delta
                
                
                
        
    def run(self, close_by_signal=False):       
        last_signal = 0
        
        equity_hist = []
        pnl_hist    = []
        
        prec_price      = 0
        cum_pnl         = 0
        prec_cum_pnl    = 0
        
        for row in self.data.iterrows():
            price = row[1]['close']
            signal = row[1]['signal']
            
            just_open = False
            
            self.pnl = 0
            
            if last_signal == 0:
                if signal == 1:
                    price = price + self.avg_ba
                    qty  = self.get_qty(price)
                    cost = self.get_costs(price, qty)
                    last_signal  = 1
                
                if signal == -1:
                    price = price - self.avg_ba
                    qty  = self.get_qty(price)
                    cost = self.get_costs(price, qty)
                    last_signal  = -1
                    
                if last_signal != 0:
                    self.equity -= cost
                    self.pnl    -= cost
                    cum_pnl -= cost
                    
                    cum_pnl = 0
                    just_open = True
                    
                    # | time open | time close | signal | entry | close | qty | type_close | pnl | pnlpct | final equity
                    j_open_time = row[1]['time']
                    j_signal = ('long' if last_signal > 0 else 'short')
                    j_entry_p = price
                    j_qty = qty
                    
            if (last_signal != 0) and (not just_open):
                pnl = self.get_pnl(prec_price, price, qty, ('long' if last_signal > 0 else 'short'))
                cum_pnl += pnl
                self.pnl += pnl
                self.equity += pnl
                
                close = False
                
                if (last_signal == 1) and (signal == 2):
                    cost = self.get_costs(price, qty)
                    self.pnl    -= cost
                    cum_pnl -= cost
                    self.equity -= cost
                    close = True
                    
                if (last_signal == -1) and (signal == -2):
                    cost = self.get_costs(price, qty)
                    self.pnl    -= cost
                    cum_pnl -= cost
                    self.equity -= cost
                    close = True
                    
                if (last_signal == -1) and (signal == 1):
                    cost = self.get_costs(price, qty) * 2
                    self.pnl    -= cost
                    cum_pnl -= cost
                    self.equity -= cost
                    close = True
                    
                if (last_signal == 1) and (signal == -1):
                    cost = self.get_costs(price, qty) * 2
                    self.pnl    -= cost
                    cum_pnl -= cost
                    self.equity -= cost
                    close = True
                
                act_close_by_signal = False
                if close_by_signal:
                    if (last_signal != 0) and (signal == 0):
                        cost = self.get_costs(price, qty)
                        self.pnl    -= cost
                        cum_pnl -= cost
                        self.equity -= cost
                        close = True
                        act_close_by_signal = True
                
                stop_loss = False
                if cum_pnl / self.equity < -self.stop_loss:
                    cost = self.get_costs(price, qty)
                    self.pnl    -= cost
                    cum_pnl -= cost
                    self.equity -= cost
                    stop_loss = True
                    close = True
                
                take_prof = False
                if not self.trail_profit[0]:
                    if cum_pnl / self.equity > self.take_profit:
                        cost = self.get_costs(price, qty)
                        self.pnl    -= cost
                        cum_pnl -= cost
                        self.equity -= cost
                        take_prof = True
                        close = True
                
                trail_prof = False
                if self.trail_profit[0]:
                    if prec_cum_pnl > 0:
                        if (cum_pnl / self.equity - prec_cum_pnl) < -self.trail_profit[1]:
                            cost = self.get_costs(price, qty)
                            self.pnl    -= cost
                            cum_pnl -= cost
                            self.equity -= cost
                            trail_prof = True
                            close = True
                            
                
                if close:
                    kind = 'close'
                    if stop_loss:
                        kind = 'stop loss'
                    if take_prof:
                        kind = 'take profit'
                    if act_close_by_signal:
                        kind = 'close by signal'
                    if trail_prof:
                        kind = 'close trail profit'
                    # | time open | time close | signal | entry | close | qty | type_close | pnl | pnlpct | final equity | dollar equity
                    j_close_time = row[1]['time']
                    j_close_p = price
                    j_kind_c = kind
                    j_pnl = cum_pnl
                    j_pnl_p = pd.np.log((cum_pnl+self.equity)/self.equity)
                    j_equity = self.equity
                    j_dol_equity = j_close_p * j_equity
                    up_jour = [j_open_time, j_close_time, j_signal, 
                               j_entry_p, j_close_p, j_qty, 
                               j_kind_c, j_pnl, j_pnl_p, j_equity, j_dol_equity]
                    self.trade_journal.append(up_jour)
                    last_signal = 0
                    cum_pnl = 0
                    
            prec_price = price
            prec_cum_pnl = cum_pnl / self.equity
                    
            equity_hist.append(self.equity)
            pnl_hist.append(self.pnl)
        
        self.data['equity'] = equity_hist
        self.data['pnl'] = pnl_hist
        labs = ['time open', 'time close', 'signal', 'price open', 'price close', 'qty', 'kind close', 'pnl', 'pnl pct', 'equity', 'dollar equity']
        self.trade_journal = pd.DataFrame(self.trade_journal, columns=labs)
        def conv_date(start, end):
            start = pd.to_datetime(start)
            end   = pd.to_datetime(end)
            diff  = end - start
            return diff.dt.total_seconds()/(60*60*24)
        self.trade_journal['duration'] = conv_date(self.trade_journal['time open'], self.trade_journal['time close'])
        self.data['dollar_equity'] = self.data['equity'] * self.data['close']
        return self.data
       
class Signal:
    def __init__(self, data):
        self.data = data
        
    def ths_signal(self, lookback_s, n_keep_pos, ths, n_min_lk=0, mean_reverse=False, allow_long=True, allow_short=True):
        
        n_lookback = len(lookback_s)
        
        def check_dats(dat):
            prices = [x for x in dat['close']]
            result = []
            
            ret = lambda x, x_1: pd.np.log(x/x_1)
            
            for l_k in lookback_s:
                up = ret(prices[-1], prices[-(l_k+1) if l_k > 1 else 2])
                
                if abs(up) > ths:
                    result.append(1)
                else:
                    result.append(0)
            
            if sum(result) >= n_min_lk:
                
                if prices[-1] > 0:
                    return 1 * factor
                if prices[-1] < 0:
                    return -1 * factor
            else:
                return 0

        factor = 1
        if mean_reverse:
            factor = -1        
        
        out_signal = []
        i = 0
        pos_counter = 0
        for row in self.data.iterrows():
            up = 0
            
            if i - n_lookback > n_lookback:
                dat = self.data.iloc[(i-max(lookback_s)-1):i]
                up = check_dats(dat)   
                
                if out_signal[-1] != 0:
                    print('\npre  ', up, out_signal[-1])
                    
                    pos_counter += 1
                    
                    if up == -out_signal[-1]:
                        pos_counter = 1
                    
                    if up == 0:
                        up = 2 * out_signal[-1]
                        pos_counter =  0
                        
                    if pos_counter >= n_keep_pos:
                        up = 2 * out_signal[-1]
                        pos_counter =  0
                        
                    print('after', up, out_signal[-1], i)
                    
                    if i > 50:
                        raise
                    
                
                if out_signal[-1] != 0:
                    if up == 0:
                        raise
                    
            out_signal.append(up)
            i += 1
            
                
        self.data['signal'] = out_signal
        return self.data    
            
        
        
    def RSI_signal(self, period, th_up, th_down, mean_reverse=False, allow_long=True, allow_short=False):
        import ta.momentum as mom
        self.data['RSI'] = mom.rsi(self.data['close'], period)
        
        factor = 1
        if mean_reverse:
            factor = -1
            
        out_signal = []
        
        for row in self.data.iterrows():
            dat = row[1]
            up = 0
            if dat['RSI'] >= th_up:
                up = 1 * factor
            if dat['RSI'] <= th_down:
                up = -1 * factor
                
            if up > 0:
                if not allow_long:
                    up = 0
            
            if up < 0:
                if not allow_short:
                    up = 0
                    
            out_signal.append(up)
            
        self.data['signal'] = out_signal
        return self.data
            
        
    
    def bband_signal(self, period, n_dev, mean_reverse=False, close_entry=False, allow_long=True, allow_short=False):
        
        import ta.volatility as vol
        import ta.momentum as mom
        self.data['up_band'] = vol.bollinger_hband(self.data['close'], period, n_dev)
        self.data['lo_band'] = vol.bollinger_lband(self.data['close'], period, n_dev)
        self.data['mean_bb'] = mom.ema(self.data['close'], period)
        
        factor = 1
        if mean_reverse:
            factor = -1        
        
        last_signal = 0
        out_signal = []
        for row in self.data.iterrows():
            dat = row[1]
            up = 0
            
            if dat['close'] > dat['up_band']:
                up = 1 * factor
            elif dat['close'] < dat['lo_band']:
                up = -1 * factor
                
            if close_entry:
                if last_signal > 0:
                    if dat['close'] < dat['mean_bb']:
                        up = 0
                if last_signal < 0:
                    if dat['close'] > dat['mean_bb']:
                        up = 0
                        
                    
            last_signal = up
            if up > 0:
                if not allow_long:
                    up = 0
            elif up < 0:
                if not allow_short:
                    up = 0
            out_signal.append(up)
        self.data['signal'] = out_signal
        return self.data        
        
    def ema_signal(self, period, ths, mean_reverse=False, close_return=False, allow_long=True, allow_short=True):
        
        import ta.momentum as mom
        self.data['ema'] = mom.ema(self.data['close'], periods=period)
        
        factor = 1
        if mean_reverse:
            factor = -1
        
        last_sig = 0
        out_signal = []
        for row in self.data.iterrows():
            dat = row[1]
            up = 0
            if dat['close'] / dat['ema'] - 1 > ths:
                up = 1 * factor
            elif dat['close'] / dat['ema'] - 1 < -ths:
                up = -1 * factor
            else:
                if close_return:
                    if last_sig == 1 * factor:
                        if dat['close'] / dat['ema'] - 1 < 0:
                            up = 2 * factor
                    elif last_sig == -1 * factor:
                        if dat['close'] / dat['ema'] - 1 > 0:
                            up = -2 * factor
                    else:
                        up = 0
            if up == 1:
                if not allow_long:
                    up = 0
            if up == -1:
                if not allow_short:
                    up = 0
            last_sig = up
            out_signal.append(up)
        self.data['signal'] = out_signal
        
        return self.data
        
    
    def bar_up_down(self, long_if, short_if, close_if, allow_long=True, allow_short=True, mean_reverse=False):
        
        ths_start = max(long_if, short_if, close_if)
        n = self.data.shape[0]
        out_signal = [0 for _ in range(ths_start)]
        last_sig = 0
        
        factor = 1
        if mean_reverse:
            factor = -1
        
        for i in range(ths_start, n):
            now = self.data.iloc[i]
            
            sig = False
            
            # check long
            if allow_long:
                if now['close'] > now['open']:
                    sig = True
                    for c in range(long_if):
                        if self.data.iloc[i-c]['close'] <= self.data.iloc[i-c-1]['close']:
                            sig = False 
                    if sig:
                        up = 1 * factor
                        last_sig = 1 * factor
                
            # check short
            if allow_short:
                if now['close'] < now['open']:  
                    sig = True              
                    for c in range(short_if):
                        if self.data.iloc[i-c]['close'] >= self.data.iloc[i-c-1]['close']:
                            sig = False    
                    if sig:
                        up = -1 * factor
                        last_sig = -1 * factor
                    
            # check close
            if last_sig != 0:
                sig = True
                
                # close long
                if last_sig > 0:
                    
                    if now['close'] < now['open']:
                        for c in range(close_if):
                            if self.data.iloc[i-c]['close'] > self.data.iloc[i-c-1]['close']:
                                sig = False
                        if sig:
                            up = 2
                            last_sig = 0
                
                # close short
                if last_sig < 0:
                    
                    if now['close'] > now['open']:
                        for c in range(close_if):
                            if self.data.iloc[i-c]['close'] < self.data.iloc[i-c-1]['close']:
                                sig = False
                        if sig:
                            up = -2
                            last_sig = 0
            if not sig:
                up = 0
            
            out_signal.append(up)
        
        if out_signal:
            self.data['signal'] = out_signal
            
        return self.data
    
    def cross_ma_signal(self, n_a, n_b, mean_reverse, allow_long, allow_short):
        '''
        n_a < n_b SEMPRE
        '''
        assert n_a < n_b, 'EMA_a < EMA_b sempre.'
        from ta.utils import ema
        self.data['ema_a'] = ema(self.data['close'], periods=n_a)
        self.data['ema_b'] = ema(self.data['close'], periods=n_b)
            
        factor = -1 if mean_reverse else 1
        
        out_signal = [0]
        n = self.data.shape[0]
            
        for i in range(1,n):
            
            up = 0
            
            prec_row = self.data.iloc[i-1]
            row = self.data.iloc[i]
            
            if row['ema_a'] > row['ema_b']:
                if prec_row['ema_a'] < row['ema_b']:
                    up = 1 * factor
            
            if row['ema_a'] < row['ema_b']:
                if prec_row['ema_a'] > row['ema_b']:
                    up = -1 * factor
            
            if up > 0:
                if not allow_long:
                    up = 0
            if up < 0:
                if not allow_short:
                    up = 0
            
            
            out_signal.append(up)
        
        self.data['signal'] = out_signal
            
            
        return self.data
            

                        
            
class FetchOrganize:
    
    def __init__(self, start, end, check_frame, trade_frame):

        self.start = start
        self.end = end
        self.check_frame = check_frame * 1000 * 60
        self.trade_frame = trade_frame * 1000 * 60
        
        self.instrument = 'BTC-PERPETUAL'
        self.convert_date_to_ts = lambda x: int(datetime.timestamp(datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))*1000)
        self.convert_ts_to_date = lambda x: datetime.strftime(datetime.fromtimestamp(x//1000), "%Y-%m-%d %H:%M:%S")
        
        self.start_ts = self.convert_date_to_ts(start)
        self.end_ts   = self.convert_date_to_ts(end)
        
        self.check_array = {'TS':list(range(self.start_ts, self.end_ts, self.check_frame))}
        
    def get_OHLC(self):
        self.trades = []
        n = len(self.check_array['TS'])
        
        off_time = 15 * 1000
        
        for i in range(n-1):
            up = PublicApi().get_prices_by_time(self.instrument, self.check_array['TS'][i]-off_time, self.check_array['TS'][i+1]-off_time, 100)
            up = [[x['timestamp'], x['timestamp']//self.trade_frame*self.trade_frame, x['price']] for x in up['trades']]
    
            for sing in up:
                self.trades.append(sing)
                
            time.sleep(0.05)
            print(self.convert_ts_to_date(self.check_array['TS'][i]))
                
        self.df_trades = pd.DataFrame(self.trades, columns=['ts', 'approx', 'price'])
        self.df_trades.sort_values('ts', inplace=True)
        
        group = self.df_trades.groupby('approx')
        _open = group.first()['price']
        _high = group.max()['price']
        _low  = group.min()['price']
        _clos = group.last()['price']
        
        out = {'time':_open.index, 'open':_open, 'high':_high, 'low':_low, 'close':_clos}
        
        df_out = pd.DataFrame(out)
        df_out.sort_index(inplace=True)
        df_out.reset_index(inplace=True)
        df_out['time'] = df_out['time'].apply(self.convert_ts_to_date)
        
        return df_out
    
    def get_heikin_ashi(self, ohlc):
        hashi = ohlc.copy()
        old_row = []
        
        start = True
        for row in hashi.iterrows():
            ind = row[0]
            dat = row[1]
            if start:
                start = False
            else:
                hashi['open'].loc[ind]  = (old_row['open'] + old_row['close']) / 2
                hashi['low'].loc[ind]   = min(hashi['open'].loc[ind], dat['open'], dat['close'], dat['low'])
                hashi['high'].loc[ind]  = max(dat['open'], dat['close'], dat['high'])
                hashi['close'].loc[ind] = (dat['open'] + dat['low'] + dat['high'] + dat['close']) / 4
            old_row = dat
        
        return hashi
    
    def run(self):
        return self.get_heikin_ashi(self.get_OHLC())
    
    
def save_file(file, save_as):
    file = file.to_csv()
    writing = open(save_as, 'w+')
    writing.write(file)
    writing.close()
    
def stats():
    t = ptf.trade_journal
    print(t)
   # print(t.groupby(['signal', 'kind close'])['pnl pct'].agg({'conta':'count', 'somma':'sum','media':'mean', 'devst':'std','max':'max', 'min':'min'}),'\n')
   
    
    
    conv = lambda x: 1 if x > 0 else 0
    hit_rate = t['pnl'].apply(conv).mean()
    
    avg_win = sum([x for x in t['pnl pct'].tolist() if x > 0]) / sum([1 for x in t['pnl pct'].tolist() if x > 0])
    avg_los = sum([x for x in t['pnl pct'].tolist() if x < 0]) / sum([1 for x in t['pnl pct'].tolist() if x < 0])
    
    ev = avg_win * hit_rate + avg_los * (1 - hit_rate)
    
    print('Hit rate    : {}%'.format(pd.np.round(100*hit_rate,2)))
    print('Avg win     : {}%'.format(pd.np.round(100*avg_win,2)))
    print('Avg loss    : {}%'.format(pd.np.round(100*avg_los,2)))
    print('Ev          : {}%'.format(pd.np.round(100*ev, 2)))
    print('Total ret   : {}%'.format(pd.np.round(100*(t['equity'].iloc[-1]/start_equity - 1), 2)))
    print('Sum profits : {}%'.format(pd.np.round(100*t['pnl'].sum()/start_equity, 2)))
    print('Dollar ret  : {}%'.format(pd.np.round(100*(result['dollar_equity'].iloc[-1]/result['dollar_equity'].iloc[0]-1), 2)))

    
def plots(bol_band, cross_ma, ema_signal):
    t = ptf.trade_journal
    long_trades = [i[0] for i in result.iterrows() if (i[1]['time'] in t['time open'].tolist()) and (i[1]['signal'] == 1)]
    short_trades = [i[0] for i in result.iterrows() if (i[1]['time'] in t['time open'].tolist()) and (i[1]['signal'] == -1)]
    #open_trades = [i[0] for i in result.iterrows() if i[1]['time'] in t['time open'].tolist()]
    close_trades = [i[0] for i in result.iterrows() if i[1]['time'] in t['time close'].tolist()]
    #ticks = [result['time'].iloc[i] for i in range(result.shape[0]) if i % 48 == 0]
    ticks = [i[1]['time'] for i in result.iterrows() if i[0] % 48 == 0]
    
    plt.subplot(2,2,1)
    plt.hist(t['pnl pct'])
    
    plt.subplot(2,2,2)
    plt.plot(result['time'], result['equity'])
    for x in long_trades:
        plt.axvline(x=x, linewidth=1, color='g')
    for x in short_trades:
        plt.axvline(x=x, linewidth=1, color='r')
    for x in close_trades:
        plt.axvline(x=x, linewidth=1, color='b')
    plt.xticks(ticks, rotation=90, fontsize=6)
    
    plt.subplot(2,2,3)
    plt.plot(result['time'], result['close'])
    if bol_band:
        plt.plot(result['time'], result['up_band'])
        plt.plot(result['time'], result['lo_band'])
    if cross_ma:
        plt.plot(result['time'], result['ema_a'])
        plt.plot(result['time'], result['ema_b'])
    if ema_signal:
        plt.plot(result['time'], result['ema'])
    for x in long_trades:
        plt.axvline(x=x, linewidth=1, color='g')
    for x in short_trades:
        plt.axvline(x=x, linewidth=1, color='r')
    for x in close_trades:
        plt.axvline(x=x, linewidth=1, color='b')
    plt.xticks(ticks, rotation=90, fontsize=6)
    
    plt.subplot(2,2,4)
    plt.plot(result['time'], result['dollar_equity'])
    plt.xticks(ticks, rotation=90, fontsize=6)

    
        
    
    
'''
data = FetchOrganize('2019-04-01 00:00:00', '2020-06-01 00:00:00', 15, 60)
ohlc = data.get_OHLC()

#ashi = data.get_heikin_ashi(ohlc)

save_file(ohlc, 'ohlc1H_eth_perp_4-19_6-20.csv')
#save_file(ashi, 'hashi1H_btc_perp_7-19_3-20.csv')

'''
ohlc = pd.read_csv('ohlc1H_btc_perp_1-19_6-20.csv')
#ashi = pd.read_csv('hashi1H_btc_perp_7-19_3-20.csv'
'''
sig = Signal(ohlc).ths_signal(lookback_s=[1], n_keep_pos=1, ths=0.02, n_min_lk=1, 
                              mean_reverse=True, allow_long=False, allow_short=True)
# ^ NON VA
'''
#sig = Signal(ohlc).ema_signal(100, ths=0.02, mean_reverse=True, close_return=False, 
#                              allow_long=True, allow_short=True)
#sig = Signal(ohlc).bar_up_down(5, 5, 2, mean_reverse=False)
#sig = Signal(ohlc).bband_signal(100, 2, mean_reverse=False, close_entry=False,
#                               allow_long=True, allow_short=True) 
'''
setup ottimali:
mean_rev=False, n/ds=100/2, delta/sl/tp=0.1/0.02/0.02
'''
sig = Signal(ohlc).cross_ma_signal(100, 200, False, True, True)
#sig = Signal(ohlc).RSI_signal(60, 60, 40, mean_reverse=False,
#                              allow_long=False, allow_short=True)

model = ANNSignal(ohlc)
model.n_batch   = 24 * 60
model.n_train   = 24 * 53
model.n_test    = 24 * 7
model.n_remodel = 24 * 7
model.use_model = 'ERF'
#sig = model.run()

ohlc['signal'] = sig['signal']

#ohlc = ohlc.iloc[-500:].reset_index()

ptf = Portfolio(ohlc)
ptf.equity          = 1
molt                = 1
ptf.delta           = 0.07 * molt
ptf.kelly_delta     = False
ptf.stop_loss_pct   = 0.04 * molt
ptf.take_profit_pct = 0.02 * molt
ptf.trail_profit    = (True, 0.02)

start_equity = ptf.equity
result = ptf.run(close_by_signal=False)

stats()
plots(bol_band=False, cross_ma=False, ema_signal=False)


end_exec = datetime.now()
diff = end_exec - start_exec

print('esecuzione in {} minuti'.format(diff.total_seconds()/60))
