# Single Hydrodynamic Field Dataloader
"""
This dataloader processes only PVI .npz files and returns a single specified hydrodynamic field
"""
####################################
## Packages
####################################
import os
import sys
import glob
import random
import typing
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath('../'))
import modules.nestedcyl_process as nc
NoneStr = typing.Union[None, str]

####################################
## DataSet Class
####################################
class PVI_SingleField_DataSet(Dataset):
	def __init__(self,
					filelist: str,
					input_field: str='rho',
					predicted: str='ptw_scale',
			        design_file: str='/data1/design_nc231213_Sn_MASTER.csv'):

		""" The definition of a dataset object for the simple nested cylinder problem:
			Nested Cylinder MOI density -> PTW scale value

			Args:
				filelist (str): Text file listing file names to read
				input_field (str): The radiographic/hydrodynamic field the model is trained on
				predicted (str): The scalar value that a model predicts
				design_file (str): .csv file with master design study parameters
		"""

		## Model Arguments 
		self.input_field = input_field
		self.predicted = predicted
		self.filelist = filelist
		self.design_file = design_file

		## Create filelist
		with open(filelist, 'r') as f:
			self.filelist = [line.rstrip() for line in f]
		self.Nsamples = len(self.filelist)

	def __len__(self):
		"""
		Return number of samples in dataset.
		"""
		return self.Nsamples

	def __getitem__(self, index):
		"""
		Return a tuple of a batch's input and output data for training at a given index.
		"""

		## Get the input image
		filepath = self.filelist[index]
		npz = np.load(filepath)
		img_input = nc.npz_pvi2field(npz, self.input_field)
		in_y, in_x = img_input.shape
		img_input = img_input.reshape((1, in_y, in_x))
		img_input = torch.tensor(img_input).to(torch.float32)

		## Get the ground truth
		key = nc.npz2key(filepath)
		truth = nc.csv2scalar(self.design_file, key, self.predicted)

		return img_input, truth
