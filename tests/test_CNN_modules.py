# TESTING FOR CNN MODULES
"""
"""

#############################
## Packages
#############################
import pytest
import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('./modules'))
import CNN_modules as CNN

#############################
## Define Fixtures
#############################
@pytest.fixture
def batchsize():
	return 8

@pytest.fixture
def H():
	return 170

@pytest.fixture
def W():
	return 50

@pytest.fixture
def kernel():
	return 5

#############################
## Test Get Conv2d Shape
#############################
class Test_GetConv2dShape():
	""" """
	def test_symmetric(self, batchsize, H, W, kernel):
		""" Tests that the get conv2d shape function correctly estimates the shape for a symmetric stride & padding """
		stride = 2
		pad = 4
		(countW, countH, countT) = CNN.conv2d_shape(w=W, h=H, k=kernel, s_w=stride, s_h=stride, p_w=pad, p_h=pad)

		conv2d = nn.Conv2d(in_channels=1,
						   out_channels=1, 
						   kernel_size=kernel, 
						   stride=stride, 
						   padding=pad,
						   padding_mode='zeros', 
						   bias=False)

		inpt = torch.rand(batchsize, 1, H, W)
		output = conv2d(inpt)
		realH, realW = output.shape[-2:]
		realT = realH * realW

		assert (countW, countH, countT) == (realW, realH, realT)

	def test_asymmetric(self, batchsize, H, W, kernel):
		""" Tests that the get conv2d shape function correctly estimates the shape for an asymmetric stride & padding """
		stride = (2, 1)
		pad = (4, 2)
		(countW, countH, countT) = CNN.conv2d_shape(w=W, h=H, k=kernel, s_w=stride[-1], s_h=stride[0], p_w=pad[-1], p_h=pad[0])

		conv2d = nn.Conv2d(in_channels=1,
						   out_channels=1, 
						   kernel_size=kernel, 
						   stride=stride, 
						   padding=pad,
						   padding_mode='zeros', 
						   bias=False)

		inpt = torch.rand(batchsize, 1, H, W)
		output = conv2d(inpt)
		realH, realW = output.shape[-2:]
		realT = realH * realW

		assert (countW, countH, countT) == (realW, realH, realT)

############################################################
## Test CNN Interpretability Module
############################################################
class Test_Interpretability_Module():
	""" """
	def test_shape(self, batchsize, H, W, kernel):
		""" Tests that the CNN Interpretability module outputs the correct shape """
		inpt = torch.rand(batchsize, 1, H, W)
		features = 12
		module = CNN.CNN_Interpretability_Module(img_size = (1, H, W),
												 kernel = kernel,
												 features = features, 
												 depth = 12,
												 act_layer = nn.GELU)
		module.eval()
		outpt = module.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, features, H, W])

############################################################
## Test CNN Reduction Module
############################################################
class Test_Reduction_Module():
	""" """
	def test_shape(self, batchsize, H, W, kernel):
		""" Tests that the CNN Reduction module outputs the correct shape """
		inpt = torch.rand(batchsize, 1, H, W)
		Hlim = 8
		Wlim = 8
		module = CNN.CNN_Reduction_Module(img_size = (1, H, W),
										 size_threshold = (Hlim, Wlim),
										 kernel = kernel,
										 stride = 2,
										 features = 12, 
										 act_layer = nn.GELU)
		module.eval()
		outpt = module.forward(inpt)
		_, _, Hout, Wout = outpt.shape
		assert (Wout <= Wlim and Hout <= Hlim)

############################################################
## Test CNN Model
############################################################
class Test_CNN_Model():
	""" """
	def test_shape(self, batchsize, H, W, kernel):
		""" Tests that the CNN Interpretability module outputs the correct shape """
		inpt = torch.rand(batchsize, 1, H, W)
		Hlim = 8
		Wlim = 8
		features = 12
		module = CNN.CNN(img_size = (1, H, W),
						 size_threshold = (Hlim, Wlim),
						 kernel = kernel,
						 features = features, 
						 interp_depth = 12,
						 act_layer = nn.GELU,
						 hidden_features = 20)
		module.eval()
		outpt = module.forward(inpt)
		assert outpt.shape == torch.Size([batchsize, 1])
