"""
Script to delete unnecessary files from a study directory

ONLY TO BE USED AFTER A STUDY IS COMPLETED

By default, this script removes:
	- all model checkpoints except the most recent
	- all .err files except the most recent (script assumes these are empty becuase slurm sucessfully continued job submission)
	- all .input files, except START.input
	- all .slurm files, except START.slurm
"""
#############################################
## Packages
#############################################
import os
import glob
import argparse

#############################################
## Inputs
#############################################
descr_str = 'Python Script to delete unnecessary files from a study directory after the study is complete'
parser = argparse.ArgumentParser(prog='Study Directory cleaning script',
								 description=descr_str,
								 fromfile_prefix_chars='@')

parser.add_argument('-SD', '--study_dir',
					action='store',
					type=str,
					default='.',
					help='Study directory to clean')

parser.add_argument('-KM', '--keep_models',
					action='store_true',
					help='Indicates if all model checkpoints should be kept; final checkpoint is always kept')


#############################################
#############################################
if __name__ == '__main__':

	args = parser.parse_args()
	study_dir = os.path.abspath(args.study_dir)
	keep_models = args.keep_models
	print('Removing unnecessary files from', study_dir)

	## Remove model checkpoints
	if not keep_models:
		print('Model checkpoint files will be removed.')
		model_files_tmpl = os.path.join(study_dir, '*_modelState_*.pth')
		model_files = sorted(glob.glob(model_files_tmpl))
		n = len(model_files)
		for i in range(n-1): #to save the last file
			os.remove(model_files[i])
		## Check that it worked
		model_files = glob.glob(model_files_tmpl)
		assert len(model_files)<=1, 'Process to remove model checkpoints has failed.'
		print('Model checkpoints have been removed, except for', model_files)
	else:
		print('Model checkpoints will NOT be removed.')

	## Remove err files
	err_files_tmpl = os.path.join(study_dir, '*.err')
	err_files = sorted(glob.glob(err_files_tmpl))
	n = len(err_files)
	for i in range(n-1): #to save the last file
		os.remove(err_files[i])
	## Check that it worked
	err_files = glob.glob(err_files_tmpl)
	assert len(err_files)==1, 'Process to remove err files has failed.'
	print('Err files have been removed, except for', os.path.basename(err_files[-1]))

	## Remove input files
	input_files_tmpl = os.path.join(study_dir, '*_restart_training_*.input')
	input_files = glob.glob(input_files_tmpl)
	for file in input_files:
		os.remove(file)
	## Check that it worked
	input_files = glob.glob(input_files_tmpl)
	assert len(input_files)==0, 'Process to remove input files has failed.'
	print('Input files have been removed')

	## Remove slurm files
	slurm_files_tmpl = os.path.join(study_dir, '*_restart_training_*.slurm')
	slurm_files = glob.glob(slurm_files_tmpl)
	for file in slurm_files:
		os.remove(file)
	## Check that it worked
	slurm_files = glob.glob(slurm_files_tmpl)
	assert len(slurm_files)==0, 'Process to remove slurm files has failed.'
	print('Slurm files have been removed')

	print('All unnecessary files have been removed')