# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:11:41 2020

@author: Ema
"""



import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import matplotlib.pyplot as plt


from utils_func import *
from out_modeling import Model



class ANNSignal:
    
    def __init__(self, ohlc):
        
        from sklearn.decomposition import PCA
        
        self.PCA = PCA
        
        self.ohlc = ohlc
        
        self.n = self.ohlc.shape[0]
        self.n_batch   = self.n
        self.n_train   = int(self.n * 0.7)
        self.n_test    = self.n - self.n_train
        self.n_remodel = int(self.n * 0.5)
        
        self.error_ratio = 1_000_000
        self.test_profit = 0.05
        self.n_retry_model = 5
        self.use_model = 'linear model'
        self.save_features = False
    
    def ta_signals(self, x_in):
        
        out = {}
        
        # 1) distanza ema
        out['ema'] = ema_distance_feature(x_in['close'], 100)
        
        # 2) distanze bol
        out['bol_a'], out['bol_b'] = bollinger_distance_feature(x_in['close'], 100, 2)
        
        # 3) rsi
        out['rsi'] = rsi_feature(x_in['close'], 100)
        
        return pd.np.array(pd.DataFrame(out))
        
    def save_array(self, name, file):
        
        pd.np.savetxt(name, file, delimiter=',')
    
    def make_x(self):
        # 1) log rendimenti laggati
        
        self.x = close2ret(self.ohlc['close'])
        self.x = row_by_n(self.x, 24)
        
        # 2) analisi tecnica
        ta_feats = self.ta_signals(self.ohlc)
        
        # fine: concatena
        self.x = pd.np.concatenate((self.x, ta_feats), axis=1)
        
        if self.save_features:
            self.save_array('x_set.csv', self.x)
        
        return self.x
        
    def make_y(self):
        self.y = close2ret(self.ohlc['close'])
        
        if self.save_features:
            self.save_array('y_set.csv', self.y)
            
    def normalize_set(self, x_train, x_test, standardization, mu_less):
        
        if mu_less:
            mus = x_train.mean()
            x_train = x_train - mus
            x_test  = x_test - mus
        
        if standardization:
            self.mus = x_train.mean()
            self.sds = x_train.std()
            x_train = (x_train - self.mus) / self.sds
            x_test  = (x_test - self.mus) / self.sds
            
        return x_train, x_test
    
    def feature_reduction(self, x_train, x_test, components):
        
        reduc_model = self.PCA(n_components=components)
        reduc_model.fit(x_train)
        x_train = reduc_model.transform(x_train)
        x_test  = reduc_model.transform(x_test)
        
        return x_train, x_test
    
    def preprocessing(self, x_train, x_test, to_standard=True, to_pca=True):
        if to_standard:
            x_train, x_test = self.normalize_set(x_train, x_test, True, False)
        if to_pca:
            x_train, x_test = self.feature_reduction(x_train, x_test, 20)
        return x_train, x_test
    
    def single_prediction(self, x_train, single_x):
        
        _, x = self.preprocessing(x_train, single_x)
        return self.signal.model.predict(x)[0]
        
    
    def cycle_signal(self):
        
        self.out_signal = pd.np.zeros(self.n)
        self.remodel_cursor = self.n_batch
        
        have_model  = False
        model_timer = 0
        
        error_ratio_check = lambda log, max_ratio: False if log['test_error'] / log['train_error'] > max_ratio else True
        test_profit_check = lambda log, min_proft: True if log['cumul return'] > min_proft else False
        
        self.models_used = []
        
        for i in range(self.n):
            
            if i > self.n_batch:
                
                if not have_model:
                
                    x_train = self.x[(i-self.n_batch):(i-self.n_test),:]
                    y_train = self.y[(i-self.n_batch):(i-self.n_test)]
                    x_test  = self.x[(i-self.n_test):i,:]
                    y_test  = self.y[(i-self.n_test):i]

                    x_train_trans, x_test_trans = self.preprocessing(x_train, x_test)
                    
                    n_tries = 0
                    condition = False
                    while not condition:
                                    
                        self.signal = Model(x_train_trans, x_test_trans, y_train, y_test, self.use_model)
                        self.signal.run()
                        
                        condition = error_ratio_check(self.signal.log_model, self.error_ratio) and test_profit_check(self.signal.log_model, self.test_profit)
                        
                        n_tries += 1
                        if n_tries > self.n_retry_model:
                            break
                        
                    if test_profit_check(self.signal.log_model, self.test_profit):
                        if error_ratio_check(self.signal.log_model, self.error_ratio):
                            have_model = True
                            self.models_used.append([i, self.signal.log_model])
                        else:
                            have_model = False
                    else:
                        have_model = False

                    
                if have_model:
                    
                    last_x  = self.x[i:(i+2),:]
                    #last_y  = self.y[i,]
                    
                    self.out_signal[i] = self.single_prediction(x_train, last_x)
                    
                    model_timer += 1
                    if model_timer > self.n_remodel:
                        have_model = False
                        model_timer = 0
            if i % 100 == 0:
                comp = pd.np.round((i/self.n)*100, 2)
                print('completion: {}%'.format(comp))
                            
        return self.out_signal
                
    def run(self):
        
        self.make_x()
        self.make_y()
        
        signal = self.cycle_signal()
        out = discretize_array(signal, 0.01)
        out = pd.DataFrame({'signal':out})
        
        return out
    
'''

ohlc = pd.read_csv('ohlc1H_btc_perp_1-19_6-20.csv')    

#ohlc = ohlc.iloc[-5000:].reset_index()  

model = ANNSignal(ohlc)

model.n_batch   = 24 * 60
model.n_train   = 24 * 53
model.n_test    = 24 * 7
model.n_remodel = 24 * 7
model.use_model = 'RF'


signal = model.run()
for i in model.models_used:
    print(i)
graf = pd.DataFrame({'test':signal, 'actual':close2ret(ohlc['close']), 'zero':pd.np.zeros(signal.shape[0])})

plt.plot(graf)
plt.legend(['test', 'actual', 'zero'])
'''