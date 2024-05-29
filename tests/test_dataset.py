# TESTING FOR NESTED CYLINDER DATASET/DATALOADER
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

sys.path.insert(0, os.path.abspath('./modules'))
sys.path.insert(0, os.path.abspath('./scripts'))
from makefilelists import maketvtlists
from pytorch_dataset import PVI_SingleField_DataSet
from pytorch_training import make_dataloader

#############################
## Define Fixtures
#############################
@pytest.fixture
def input_dir():
	return './data/'

@pytest.fixture
def search_dir(input_dir):
	return os.path.join(input_dir, '*pvi*.npz')

@pytest.fixture
def design_file():
	return './data/nestedcyl_design_file.csv'

@pytest.fixture
def batch_size():
	return 8

@pytest.fixture
def num_batches():
	return 10

def removefile(file:str):
	""" Checks if file exists, and removed it if so """
	if os.path.isfile(file): os.remove(file)

#############################
## Test NestedCyl Dataset
#############################
class Test_PVI_SingleField_DataSet():
	""" """
	def test_givefilelist(self, search_dir, design_file):
		""" Tests that dataset generates with a given filelist """
		maketvtlists(search_dir = search_dir, 
					sample_split = (0.1, 0, 0),
					save_path = './pytest',
					save = True)
		try:
			PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
							input_field = 'rho',
							predicted = 'ptw_scale',
							design_file = design_file)
			removefile('./pytest_train_samples.txt')
		except:
			assert False, "PVI_SingleField_DataSet raises error when passed filelist argument."
		else:
			assert True

	def test_len(self, search_dir, design_file):
		""" Tests that dataset.__len__() operates as intended """
		train_samples, _, _ = maketvtlists(search_dir = search_dir, 
											sample_split = (0.1, 0, 0),
											save_path = './pytest',
											save = True)
		dataset = PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
									input_field = 'rho',
									predicted = 'ptw_scale',
									design_file = design_file)
		removefile('./pytest_train_samples.txt')
		assert dataset.__len__() == len(train_samples)

	def test_getimage(self, search_dir, design_file):
		""" Tests that dataset.__getitem__() returs a correcty sized image """
		maketvtlists(search_dir = search_dir, 
						sample_split = (0.1, 0, 0),
						save_path = './pytest',
						save = True)
		dataset = PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
									input_field = 'rho',
									predicted = 'ptw_scale',
									design_file = design_file)
		removefile('./pytest_train_samples.txt')
		img_input, truth = dataset.__getitem__(1)
		assert img_input.shape == torch.Size([1, 1700, 500])

	def test_gettruth(self, search_dir, design_file):
		""" Tests that dataset.__getitem__() returs a correcty sized truth vector """
		maketvtlists(search_dir = search_dir, 
						sample_split = (0.1, 0, 0),
						save_path = './pytest',
						save = True)
		dataset = PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
									input_field = 'rho',
									predicted = 'ptw_scale',
									design_file = design_file)
		removefile('./pytest_train_samples.txt')
		img_input, truth = dataset.__getitem__(1)
		assert isinstance(truth, float)

#############################
## Test NestedCyl Dataloader
#############################
class Test_nestedcyldataloader():
	""" """
	def test_batchsize(self, search_dir, design_file, batch_size, num_batches):
		""" Tests that dataloader returns an item of the correct batchsize """
		maketvtlists(search_dir = search_dir, 
						sample_split = (0.1, 0, 0),
						save_path = './pytest',
						save = True)
		dataset = PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
									input_field = 'rho',
									predicted = 'ptw_scale',
									design_file = design_file)
		dataloader = make_dataloader(dataset,
											batch_size = batch_size,
											num_batches = num_batches)
		removefile('./pytest_train_samples.txt')
		assert len(next(iter(dataloader))[0]) == batch_size

	def test_num_batches(self, search_dir, design_file, batch_size, num_batches):
		""" Tests that dataloader returns the correct number of batches """
		num_batches = 10
		maketvtlists(search_dir = search_dir, 
						sample_split = (0.1, 0, 0),
						save_path = './pytest',
						save = True)
		dataset = PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
									input_field = 'rho',
									predicted = 'ptw_scale',
									design_file = design_file)
		dataloader = make_dataloader(dataset,
											batch_size = batch_size,
											num_batches = num_batches)
		removefile('./pytest_train_samples.txt')
		assert len(dataloader) == num_batches

	def test_imgshape(self, search_dir, design_file, batch_size, num_batches):
		""" Tests that dataloader returns an image of the correct shape """
		maketvtlists(search_dir = search_dir, 
						sample_split = (0.1, 0, 0),
						save_path = './pytest',
						save = True)
		dataset = PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
									input_field = 'rho',
									predicted = 'ptw_scale',
									design_file = design_file)
		dataloader = make_dataloader(dataset,
											batch_size = batch_size,
											num_batches = num_batches)
		removefile('./pytest_train_samples.txt')
		img_input, truth = next(iter(dataloader))
		assert img_input.shape == torch.Size([batch_size, 1, 1700, 500])

	def test_truthshape(self, search_dir, design_file, batch_size, num_batches):
		""" Tests that dataloader returns an image of the correct shape """
		maketvtlists(search_dir = search_dir, 
						sample_split = (0.1, 0, 0),
						save_path = './pytest',
						save = True)
		dataset = PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
									input_field = 'rho',
									predicted = 'ptw_scale',
									design_file = design_file)
		dataloader = make_dataloader(dataset,
											batch_size = batch_size,
											num_batches = num_batches)
		removefile('./pytest_train_samples.txt')
		img_input, truth = next(iter(dataloader))
		assert truth.shape == torch.Size([batch_size])

	def test_random_shuffle(self, search_dir, design_file, num_batches):
		""" Tests that dataloader will shuffle data from an input list """
		maketvtlists(search_dir = search_dir, 
						sample_split = (0.1, 0, 0),
						save_path = './pytest',
						save = True)
		dataset = PVI_SingleField_DataSet(filelist = './pytest_train_samples.txt',
									input_field = 'rho',
									predicted = 'ptw_scale',
									design_file = design_file)
		dataloader1 = make_dataloader(dataset,
											batch_size = 1,
											num_batches = num_batches)
		dataloader2 = make_dataloader(dataset,
											batch_size = 1,
											num_batches = num_batches)
		removefile('./pytest_train_samples.txt')

		dataiter1 = iter(dataloader1)
		dataiter2 = iter(dataloader2)
		truth_array = []
		for b in range(num_batches):
			item1 = next(dataiter1)[0]
			item2 = next(dataiter2)[0]
			match = torch.equal(item1, item2)
			truth_array.append(match)

		assert not np.all(truth_array)
