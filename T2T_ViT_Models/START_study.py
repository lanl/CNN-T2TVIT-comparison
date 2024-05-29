""" Script to start training a study
"""

####################################
## Packages
####################################
import os
import sys
import argparse
import numpy as np
import pandas as pd
import shutil

####################################
## Helper Function
####################################
def replace_keys(study_dict: dict, data:str):
	""" Function to replace "key" values in a string with dictionary values
		Args:
			study_dict (dict): dictonary of keys and values to replace
			data (str): data to replace keys in
		Returns:
			data (str): data with keys replaced
	"""
	for key, value in study_dict.items():
		if type(value)==np.float64 or type(value)==float:
			data = data.replace(f'<{key}>', '{:5.4f}'.format(value))
		elif type(value)==np.int64 or type(value)==int:
			data = data.replace(f'<{key}>', '{:d}'.format(value))
		elif type(value)==str:
			data = data.replace(f'<{key}>', '{}'.format(value))
		elif type(value)==np.bool_ or type(value)==bool:
			data = data.replace(f'<{key}>', '{}'.format(str(value)))
		else:
			print('Key is', key, 'with value of', value, 'with type', type(value))
			raise ValueError('Unrecognized datatype in hyperparameter list.')

	return data

####################################
## Process Hyperparameters
####################################
## .csv argparse argument
descr_str = 'Starts execution of T2TViT training'
parser = argparse.ArgumentParser(prog='SJC-T2TViT',
								 description=descr_str)
parser.add_argument('--csv',
					action='store',
					type=str,
					default='./hyperparameters.csv',
					help='CSV file containing study hyperparameters')
args = parser.parse_args()

training_input_tmpl = './training_input.tmpl'
training_slurm_tmpl = './training_slurm.tmpl'
training_START_input = './training_START.input'
training_START_slurm = './training_START.slurm'

## Process Hyperparmaeters File
studyDF = pd.read_csv(args.csv,
					  sep=',',
					  header=0,
					  index_col=0,
					  comment='#',
					  engine='python')
varnames = studyDF.columns.values
idxlist = studyDF.index.values

## Save Hyperparameters to Dictionary List
studylist = []
for i in idxlist:
	studydict = {}
	studydict['studyIDX'] = int(i)
	for var in varnames:
		studydict[var] = studyDF.loc[i, var]
	studylist.append(studydict)

####################################
## Run Studies
####################################
## Iterate Through Dictionary List to Run Studies
for k, study in enumerate(studylist):

	## Make Study Directory
	studydirname = 'study_{:02d}'.format(study['studyIDX'])
	if not os.path.exists(studydirname):
		os.makedirs(studydirname)

	## Make new training_input.tmpl file
	with open(training_input_tmpl, 'r') as f:
		training_input_data = f.read()
	training_input_data = replace_keys(study, training_input_data)
	training_input_filepath = os.path.join(studydirname, 'training_input.tmpl')
	with open(training_input_filepath, 'w') as f:
		f.write(training_input_data)

	## Make new training_slurm.tmpl file
	with open(training_slurm_tmpl, 'r') as f:
		training_slurm_data = f.read()
	training_slurm_data = training_slurm_data.replace('<studyIDX>', '{:02d}'.format(study['studyIDX']))        
	training_slurm_filepath = os.path.join(studydirname, 'training_slurm.tmpl')
	with open(training_slurm_filepath, 'w') as f:
		f.write(training_slurm_data)

	## Make new training_START.input file
	with open(training_START_input, 'r') as f:
		START_input_data = f.read()
	START_input_data = replace_keys(study, START_input_data)
	START_input_name = 'study{:02d}_START.input'.format(study['studyIDX'])
	START_input_filepath = os.path.join(studydirname, START_input_name)
	with open(START_input_filepath, 'w') as f:
		f.write(START_input_data)

	## Make a new training_START.slurm file
	with open(training_START_slurm, 'r') as f:
		START_slurm_data = f.read()
	START_slurm_data = START_slurm_data.replace('<studyIDX>', '{:02d}'.format(study['studyIDX']))        
	START_slurm_name = 'study{:02d}_START.slurm'.format(study['studyIDX'])
	START_slurm_filepath = os.path.join(studydirname, START_slurm_name)
	with open(START_slurm_filepath, 'w') as f:
		f.write(START_slurm_data)

	## Submit Job
	os.system(f'cd {studydirname}; sbatch {START_slurm_name}; cd ..')
