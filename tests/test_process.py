# TESTING FOR NESTED CYLINDER DATA PROCESSING FUNCTIONS
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

sys.path.insert(0, os.path.abspath('./modules'))
import nestedcyl_process as nc

#############################
## Define Fixtures
#############################
@pytest.fixture
def npz_pvi_file():
	return './data/nc231213_Sn_id0020_pvi_idx00112.npz'

@pytest.fixture
def design_file():
	return './data/nestedcyl_design_file.csv'

@pytest.fixture
def field():
	return 'rho'


#############################
## Test npz2key Function
#############################
class Test_npz2key():
	""" """
	def test_pvi(self, npz_pvi_file):
		""" Tests that the npz2key function identifies the key correctly from a PVI file """
		assert nc.npz2key(npz_pvi_file) == 'nc231213_Sn_id0020'

#############################
## Test csv2scalePTW Function
#############################
class Test_csv2scalar():
	""" """
	def test_pvi_ptw(self, npz_pvi_file, design_file):
		""" Tests that the csv2scalar function identifies the PTW scale constant correctly from a PVI file """
		key = nc.npz2key(npz_pvi_file)
		assert nc.csv2scalar(design_file, key, 'ptw_scale') ==   0.9213666069498454

	def test_invalid_scalar(self, npz_pvi_file, design_file):
		""" Tests that the csv2scalar function raises an error when the scalar name does not apper in the design file """
		key = nc.npz2key(npz_pvi_file)
		with pytest.raises(AssertionError):
			value = nc.csv2scalar(design_file, key, 'invalid')


#############################
## Test get_field Function
#############################
class Test_get_field():
	""" """
	def test_shape(self, npz_pvi_file, field):
		""" Tests that the get_field function returns the correct shape """
		npz = np.load(npz_pvi_file)
		assert nc.npz_pvi2field(npz, field).shape == (1700, 500)