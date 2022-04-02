# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 20:50:33 2022

@author: cansu
"""

import numpy as np
import ActivationFunctions as af


def backpropagation_algorithm_with_sigmoid(p,t):
    
    p_rows, p_cols = p.shape
    t_rows, t_cols = t.shape
    
    input_size,output_size,hidden_size, = p_cols, t_cols, 3
    learning_rate = 0.1
    
    
    
    # Fill hidden and output layers with random values.
    
    w_hidden = np.random.uniform(size=(input_size, hidden_size))
    b_hidden = np.random.uniform(size=(1, hidden_size))*np.ones([p_rows, hidden_size])
    w_output = np.random.uniform(size=(hidden_size, output_size))
    b_output = np.random.uniform(size=(1, output_size))*np.ones([p_rows, output_size])
    
    
    # Learning iteration
    while True:
        # Forward propagation
        actual_hidden = af.sigmoid(np.dot(p, w_hidden)) +b_hidden
        output = np.dot(actual_hidden, w_output) +b_output
    
        # Calculate error (expected output - calculated output)
        error = t - output
        
        # Backward Propagation
        dZ = error * learning_rate
        w_output += actual_hidden.T.dot(dZ)
        b_output += dZ
    
        dH = dZ.dot(w_output.T) * af.sigmoid_prime(actual_hidden)
        w_hidden += p.T.dot(dH)
        b_hidden += dH
        
        #when total error < 0.01 break
        total_error=0
        for x in error:
            total_error +=abs(x)
        
        if total_error<0.01:
          break;
          
    for i in range(p_rows):
        actual_hidden = af.sigmoid(np.dot(p[i], w_hidden))+b_hidden[i]
        actual_output = np.dot(actual_hidden, w_output)+b_output[i]
        print(p[i], actual_output,"expected output",t[i])
    
    
    
    
def backpropagation_algorithm_with_tanh(p,t):
    
    p_rows, p_cols = p.shape
    t_rows, t_cols = t.shape
    
    input_size,output_size,hidden_size, = p_cols, t_cols, 3
    learning_rate = 0.1
    
    
    
    # Fill hidden and output layers with random values.
    
    w_hidden = np.random.uniform(size=(input_size, hidden_size))
    b_hidden = np.random.uniform(size=(1, hidden_size))*np.ones([p_rows, hidden_size])
    w_output = np.random.uniform(size=(hidden_size, output_size))
    b_output = np.random.uniform(size=(1, output_size))*np.ones([p_rows, output_size])
    
    
    # Learning iteration
    while True:
        # Forward propagation
        actual_hidden = af.tanh(np.dot(p, w_hidden)) +b_hidden
        output = np.dot(actual_hidden, w_output) +b_output
    
        # Calculate error (expected output - calculated output)
        error = t - output
        
        # Backward Propagation
        dZ = error * learning_rate
        w_output += actual_hidden.T.dot(dZ)
        b_output += dZ
    
        dH = dZ.dot(w_output.T) * af.tanh_prime(actual_hidden)
        w_hidden += p.T.dot(dH)
        b_hidden += dH
        
        #when total error < 0.01 break
        total_error=0
        for x in error:
            total_error +=abs(x)
        
        if total_error<0.01:
          break;
          
    for i in range(p_rows):
        actual_hidden = af.tanh(np.dot(p[i], w_hidden))+b_hidden[i]
        actual_output = np.dot(actual_hidden, w_output)+b_output[i]
        print(p[i], actual_output,"expected output",t[i])
    