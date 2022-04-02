# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:45:46 2022

@author: cansu
"""

import numpy as np

def sigmoid(x):
    
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1.0 - x)

def tanh(x):

    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def tanh_prime(x):
  
    return (1.0 - x ** 2.0)