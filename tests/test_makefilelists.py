# TESTING FOR MAKEFILELIST FUNCTIONS
"""
This test suite needs to be run on dodona, where these is acess to the nested cylinder .npz files
"""

#############################
## Packages
#############################
import pytest
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath('./scripts'))
from makefilelists import findcorruptedfiles, maketvtlists

#############################
## Define Fixtures
#############################
@pytest.fixture
def input_dir():
	return '/data/'

@pytest.fixture
def design_file():
	return '/data/nestedcyl_design_file.csv'

def removefile(file:str):
	""" Checks if file exists, and removed it if so """
	if os.path.isfile(file): os.remove(file)

#######################################
## Test findcorruptedfiles Function
#######################################
class Test_findcorruptedfiles():
	""" """
	def runs(self, input_dir):
		""" Tests function runs without error """
		trainlist, _, _ = ncdata = maketvtlists(search_dir = input_dir+'*.npz',
												sample_split = (1.0, 0, 0), 
												save_path = None, 
												save = False)
		try:
			corrupted = findcorruptedfiles(input_dir = input_dir, 
											samplelist = trainlist)
		except:
			assert False, "findcorruptedfiles function raises error."
		else:
			assert True


#############################
## Test maketvtlists Function
#############################
class Test_maketvtlists():
	""" """
	def test_invalidsplit(self, input_dir):
		""" Tests that maketvtlists function gives an assertion error when passed an invalid sample_split """
		with pytest.raises(AssertionError):
			maketvtlists(search_dir = input_dir+'*.npz', 
						sample_split = (1, 1, 1),
						save_path = None,
						save = False)


	def test_saveNone(self, input_dir):
		""" Tests that maketvtlists function gives an assertion error when save=True and save_path=None """
		with pytest.raises(ValueError):
			maketvtlists(search_dir = input_dir+'*.npz', 
						sample_split = (1, 0, 0),
						save_path = None,
						save = True)

	def test_trainzero(self, input_dir):
		""" Tests that val_samples is empty and not saved to file when val=0 """
		train_samples, val_samples, test_samples = maketvtlists(search_dir = input_dir+'*.npz', 
																sample_split = (0, 1, 0),
																save_path = './pytest',
																save = True)
		train_filelist = os.path.isfile('./pytest_train_samples.txt')

		## Remove created files
		removefile('./pytest_train_samples.txt')
		removefile('./pytest_val_samples.txt')
		removefile('./pytest_test_samples.txt')

		assert train_samples==[] and not train_filelist, "When train=0, train_samples is nonempty."

	def test_valzero(self, input_dir):
		""" Tests that val_samples is empty and not saved to file when val=0 """
		train_samples, val_samples, test_samples = maketvtlists(search_dir = input_dir+'*.npz', 
																sample_split = (1, 0, 0),
																save_path = './pytest',
																save = True)
		val_filelist = os.path.isfile('./pytest_val_samples.txt')

		## Remove created files
		removefile('./pytest_train_samples.txt')
		removefile('./pytest_val_samples.txt')
		removefile('./pytest_test_samples.txt')

		assert val_samples==[] and not val_filelist, "When val=0, val_samples is nonempty."

	def test_testzero(self, input_dir):
		""" Tests that test_samples is empty and not saved to file when test=0 """
		train_samples, val_samples, test_samples = maketvtlists(search_dir = input_dir+'*.npz', 
																sample_split = (1, 0, 0),
																save_path = './pytest',
																save = True)
		test_filelist = os.path.isfile('./pytest_test_samples.txt')

		## Remove created files
		removefile('./pytest_train_samples.txt')
		removefile('./pytest_val_samples.txt')
		removefile('./pytest_test_samples.txt')

		assert test_samples==[] and not test_filelist, "When test=0, test_samples is nonempty."

	def test_savefiles(self, input_dir):
		""" Tests that all three sample lists are saved when they are nonempty """
		train_samples, val_samples, test_samples = maketvtlists(search_dir = input_dir+'*', 
																sample_split = (0.7, 0.2, 0.1),
																save_path = './pytest',
																save = True)

		train_filelist = os.path.isfile('./pytest_train_samples.txt')
		val_filelist = os.path.isfile('./pytest_val_samples.txt')
		test_filelist = os.path.isfile('./pytest_test_samples.txt')

		## Remove created files
		removefile('./pytest_train_samples.txt')
		removefile('./pytest_val_samples.txt')
		removefile('./pytest_test_samples.txt')

		assert train_filelist and val_filelist and test_filelist, "Not all sample lists are saved to file."
