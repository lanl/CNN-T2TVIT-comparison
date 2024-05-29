# NESTED CYLINDER DATA PROCESSING
"""
Contains functions that process the .npz files for the nested cylinder dataset

Note that some .npz file names contain the "pvi" flag and contain hydrodynamic fields
and some .npz file names contain the "pdv" and contain the photon doppler velocimetry traces
"""
####################################
## Packages
####################################
import numpy as np
import pandas as pd

#####################################
## Functions
#####################################
def npz2key(npz_file: str):
	""" Function to extract study information from the name of an .npz file

	    Args:
	        npz_file (str): file path from working directory to .npz file
	    
	    Returns:
			key (str): 	the study information for the simulation that generated the .npz file; of the form "nc231213_Sn_id####"
    """
	key = npz_file.split('/')[-1].split('_')
	key = '_'.join(key[0:3])
	return key

def csv2scalar(csv_file: str, key:str, scalar:str):
	""" Function to extract the dcj value from the design .csv file given the study key
	    
	    Args:
	        csv_file (str): file path from working directory to the .csv design file
	        key (str): 	the study information for a given simulation; of the form "nc231213_Sn_id####"
	        scalar (str): name of scalar to extract from the design file
	    
	    Returns:
			value (float): the value of the scalar for the specified key
    """
	design_df = pd.read_csv(csv_file,
								sep=',',
                            	header=0,
                            	index_col=0,
                            	engine='python')
	#removed spaces from headers
	for col in design_df.columns:
		design_df.rename(columns={col: col.strip()}, inplace=True)
	assert scalar in design_df.columns, "csv2scalar: selected scalar is not in the design file"
	value = design_df.at[key, scalar]
	return value

def npz_pvi2field(npz: np.lib.npyio.NpzFile, field: str):
	""" Function to extract a field "picture" array from an .npz file

	    Args:
	        npz (np.lib.npyio.NpzFile): a loaded .npz file
	        field (str): name of field to extract

	    Returns:
			pic (np.ndarray[(1700, 500), float]): field 
    """
	pic = npz[field]
	pic = pic[800:, :250]
	pic = np.concatenate((np.fliplr(pic), pic), axis=1)
	return pic
