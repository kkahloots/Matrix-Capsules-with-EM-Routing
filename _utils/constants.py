# -*- coding: utf-8 -*-

VAE = 0
VAECNN = 1

GMVAE = 2
GMVAECNN = 3

VMA = 4
VMACNN = 5

GME = 6
GMECNN = 7

# Stopping tolerance
tol = 1e-6

convA = 32 # number of channels in output from ReLU Conv1
capB = 8 #number of capsules in output from PrimaryCaps
capC = 16 #number of channels in output from ConvCaps1
capD = 16 #number of channels in output from ConvCaps2

m_schedule = 0.2 #the m will get to 0.9 at current epoch
iter_routing = 3  #number of iterations

ac_lambda0 = 0.01 #lambda in the activation function a_c, iteration 0
ac_lambda_step = 0.01 #It is described that lambda increases at each iteration with a fixed schedule, however specific super parameters is absent.


epsilon = 1e-9 #epsilon
m_plus = 0.9 #the parameter of m plus
m_minus =  0.1 #the parameter of m minus
lambda_val = 0.5 #down weight of the loss for absent digit classes

