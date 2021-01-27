# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:51:36 2020

@author: Ema
"""


import pandas as pd
from utils_func import discretize_array, make_random_state


class Model:
    
    def __init__(self, x_train, x_test, y_train, y_test, use_model):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test
        self.use_model = use_model
        
        self.log_model = {}
        
        if self.use_model == 'LM':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        
        elif self.use_model == 'MLP':
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(activation='tanh', solver='sgd', 
                                      hidden_layer_sizes=(10,6,3), max_iter=300,
                                      random_state=make_random_state())
            
            self.log_model['random state'] = self.model.random_state
            
        elif self.use_model == 'RF':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(random_state=make_random_state())
            self.log_model['random state'] = self.model.random_state
            
        elif self.use_model == 'ERF':
            from sklearn.ensemble import ExtraTreesRegressor
            self.model = ExtraTreesRegressor(random_state=make_random_state())
            self.log_model['random state'] = self.model.random_state
            
        elif self.use_model == 'SGD':
            from sklearn.linear_model import SGDRegressor
            self.model = SGDRegressor(random_state=make_random_state())
            self.log_model['random state'] = self.model.random_state

        self.log_model['kind'] = self.use_model

            
    def fit_predict(self):
        self.model.fit(self.x_train, self.y_train)
        self.on_train_prediction = self.model.predict(self.x_train)
        self.prediction = self.model.predict(self.x_test)
        
    def single_predict(self, x):
        return self.model.predict(x)[0]
        
    def calculate_errors(self, methods):
        
        if 'square errors' in methods:
            
            sq_err_func = lambda y_test, y_out: sum([(y_out[i]-y_test[i])**2 for i in range(y_test.shape[0])]) / y_test.shape[0]
            
            train_error = sq_err_func(self.y_train, self.on_train_prediction)
            test_error  = sq_err_func(self.y_test, self.prediction)
            
            self.log_model['train_error'] = train_error
            self.log_model['test_error']  = test_error
            
        if 'cum return' in methods:
            
            weights = discretize_array(self.prediction, 0)
            self.model_return = pd.np.cumsum(weights * self.y_test)
            self.log_model['cumul return'] = self.model_return[-1]
                    
    def run(self):
        
        self.fit_predict()
        self.calculate_errors(['square errors', 'cum return'])
