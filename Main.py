# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 21:11:10 2022

@author: cansu
"""

import numpy as np
import BackpropagationAlgorithm as bpa
p1 = np.array([[2, 2],[1,-2],[-2,2],[1,1]])
t1 = np.array([[0], [1], [0], [1]])

p2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t2 = np.array([[0], [0], [0], [1]])

p3 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t3 = np.array([[0], [1], [1], [1]])

p4 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t4 = np.array([[0], [1], [1], [0]]) 

p5 = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
t5 = np.array([[-1], [1], [1], [-1],[1], [-1], [-1], [1]])



print("TEST")
bpa.backpropagation_algorithm_with_sigmoid(p1, t1)
print("-------------------------------------")

print("AND")
bpa.backpropagation_algorithm_with_sigmoid(p2, t2)
print("-------------------------------------")

print("OR")
bpa.backpropagation_algorithm_with_sigmoid(p3, t3)
print("-------------------------------------")

print("XOR")
bpa.backpropagation_algorithm_with_sigmoid(p4, t4)
print("-------------------------------------")

print("Parite 3")
bpa.backpropagation_algorithm_with_tanh(p5, t5)
print("-------------------------------------")



