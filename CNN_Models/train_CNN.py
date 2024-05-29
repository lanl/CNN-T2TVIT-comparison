# TRAINING SCRIPT FOR PVI SINGLE-FIELD CNN
"""

"""
#############################################
## Packages
#############################################
import sys
import os
import typing
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
from modules.CNN_modules import CNN
from modules.pytorch_dataset import PVI_SingleField_DataSet
import modules.pytorch_training as tr

#############################################
## Inputs
#############################################
descr_str = 'Trains a PVI Single Field CNN to estimate PTW strength scale'
parser = argparse.ArgumentParser(prog='PVI Single Field CNN Training',
								 description=descr_str,
								 fromfile_prefix_chars='@')

#############################################
## Learning Problem
#############################################
parser.add_argument('--studyIDX',
					action='store',
					type=int,
					default=1,
					help='Study ID number to match hyperparameters')

parser.add_argument('--input_field',
					action='store',
					type=str,
					default='hr_MOICyl',
					help='Data field the models will train on')

parser.add_argument('--predicted_scalar',
					action='store',
					type=str,
					default='ptw_scale',
					help='Scalar value the models will predict')

parser.add_argument('--interp_depth',
					action='store',
					type=int,
					default=12,
					help='Number of interpertability blocks in the model')

#############################################
## File Paths
#############################################
parser.add_argument('--design_file',
					action='store',
					type=str,
					default='/data1/design_nc231213_Sn_MASTER.csv',
					help='.csv file that contains the truth values for data files')


#############################################
## Training Parameters
#############################################
parser.add_argument('--initialization_seed',
					action='store',
					type=float,
					default=1.0,
					help='Seed for the initialization of model weights')

parser.add_argument('--init_learnrate',
					action='store',
					type=float,
					default=1e-3,
					help='Initial learning rate')

parser.add_argument('--batch_size',
					action='store',
					type=int,
					default=64,
					help='Batch size')

#############################################
## Epoch Parameters
#############################################
parser.add_argument('--total_epochs',
					action='store',
					type=int,
					default=10,
					help='Total training epochs')

parser.add_argument('--cycle_epochs',
					action='store',
					type=int,
					default=5,
					help='Number of epochs between saving the model and re-quequing training process; must be able to be completed in the set wall time')

parser.add_argument('--train_batches',
					action='store',
					type=int,
					default=250,
					help='Number of batches to train on in a given epoch')

parser.add_argument('--val_batches',
					action='store',
					type=int,
					default=25,
					help='Number of batches to validate on in a given epoch')

parser.add_argument('--continuation',
					action='store_true',
					help='Indicates if training is being continued or restarted')

parser.add_argument('--checkpoint',
					action='store',
					type=str,
					default='None',
					help='Path to checkpoint to continue training from')

#############################################
#############################################
if __name__ == '__main__':

	#############################################
	## Process Inputs
	#############################################
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()

	## Study ID
	studyIDX = args.studyIDX

	## Data Paths
	input_field = args.input_field
	predicted_scalar = args.predicted_scalar
	depth = args.interp_depth
	design_file = os.path.abspath(args.design_file)
	seed = args.initialization_seed

	## Training Parameters
	initial_learningrate = args.init_learnrate
	batch_size = args.batch_size

	## Epoch Parameters
	total_epochs = args.total_epochs
	cycle_epochs = args.cycle_epochs
	train_batches = args.train_batches
	val_batches = args.val_batches
	CONTINUATION = args.continuation
	START = not CONTINUATION
	checkpoint = args.checkpoint

	#############################################
	## Check Devices
	#############################################
	print('\n')
	print('Slurm & Device Information')
	print('=========================================')
	print('Slurm Job ID:', os.environ['SLURM_JOB_ID'])
	print('Pytorch Cuda Available:', torch.cuda.is_available())
	print('GPU ID:', os.environ['SLURM_JOB_GPUS'])
	print('Number of System CPUs:', os.cpu_count())
	print('Number of CPUs per GPU:', os.environ['SLURM_JOB_CPUS_PER_NODE'])

	print('\n')
	print('Model Training Information')
	print('=========================================')


	#############################################
	## Initialize Model
	#############################################
	torch.manual_seed(seed)
	model = CNN(img_size = (1, 1700, 500),
				size_threshold = (8, 8),
				kernel = 5,
				features = 12, 
				interp_depth = depth,
				act_layer = nn.GELU,
				hidden_features = 20)
	model.to(device)

	#############################################
	## Initialize Optimizer
	#############################################
	optimizer = torch.optim.AdamW(model.parameters(),
								lr = initial_learningrate,
								betas = (0.9, 0.999),
								eps = 1e-08,
								weight_decay = 0.01)

	#############################################
	## Initialize Loss
	#############################################
	loss_fn = nn.MSELoss(reduction='none')

	print('Model initialized.')

	#############################################
	## Load Model for Continuation
	#############################################
	if CONTINUATION:
		checkpoint = torch.load(checkpoint, map_location=device)
		starting_epoch = checkpoint["epoch"]
		model.load_state_dict(checkpoint["modelState"])
		optimizer.load_state_dict(checkpoint["optimizerState"])
		print('Model state loaded for continuation.')
	else:
		starting_epoch = 0

	#############################################
	## Initialize Data
	#############################################
	train_filelist = os.path.join('..', '..', 'filelists', 'PVI_only_train_samples.txt')
	val_filelist = os.path.join('..', '..', 'filelists', 'PVI_only_val_samples.txt')
	test_filelist = os.path.join('..', '..', 'filelists', 'PVI_only_test_samples.txt')

	train_dataset = PVI_SingleField_DataSet(filelist = train_filelist,
									input_field = input_field,
									predicted = predicted_scalar,
									design_file = design_file)
	val_dataset = PVI_SingleField_DataSet(filelist = val_filelist,
									input_field = input_field,
									predicted = predicted_scalar,
									design_file = design_file)
	test_dataset = PVI_SingleField_DataSet(filelist = test_filelist,
									input_field = input_field,
									predicted = predicted_scalar,
									design_file = design_file)

	print('Datasets initialized.')

	#############################################
	## Training Loop
	#############################################
	##Initialize Dictionaries
	train_val_summary_dict = {
							"train_loss": [],
							"val_loss": [],
							"epoch_time": []
							}
	train_samples_dict = {
							"epoch": [],
							"batch": [],
							"truth": [],
							"prediction": [],
							"loss": []
							}
	val_samples_dict = {
							"epoch": [],
							"batch": [],
							"truth": [],
							"prediction": [],
							"loss": []
							}

	## Train Model
	print("Training Model . . .")
	starting_epoch += 1
	ending_epoch = min(starting_epoch+cycle_epochs, total_epochs+1)

	for e in range(starting_epoch, ending_epoch):
		## Setup Dataloaders
		train_dataloader = tr.make_dataloader(train_dataset, batch_size, train_batches)
		val_dataloader = tr.make_dataloader(val_dataset, batch_size, val_batches)

		## Train an Epoch
		train_val_summary_dict, train_samples_dict, val_samples_dict = tr.train_epoch(training_data = train_dataloader,
																						validation_data = val_dataloader, 
																						model = model,
																						optimizer = optimizer,
																						loss_fn = loss_fn,
																						summary_dict = train_val_summary_dict,
																						train_sample_dict = train_samples_dict,
																						val_sample_dict = val_samples_dict,
																						device = device)

		## Add Epoch Info to Dicts
		#### this works by replacing "epoch zero" with the correct epoch
		train_samples_dict["epoch"] = [x or e for x in train_samples_dict["epoch"]]
		val_samples_dict["epoch"] = [x or e for x in val_samples_dict["epoch"]]

		## Print Summary Results
		print('Completed epoch '+str(e)+':')
		print('\tTraining Loss:', train_val_summary_dict["train_loss"][-1])
		print('\tValidation Loss:', train_val_summary_dict["val_loss"][-1])
		print('\tEpoch Time:', train_val_summary_dict["epoch_time"][-1])

	## Save Model Checkpoint
	print("Saving model checkpoint at end of epoch "+ str(e) + ". . .")
	new_checkpoint = {
					"epoch"          : e,
					"modelState"     : model.state_dict(),
					"optimizerState" : optimizer.state_dict()
					}
	new_checkpoint_path = os.path.join('./', 'study{0:02d}_modelState_epoch{1:03d}.pth'.format(studyIDX, e))
	torch.save(new_checkpoint, new_checkpoint_path)
	print("Model checkpoint saved at end of epoch "+ str(e) + ".")


	## Save Summary Dictionary
	summarydf = pd.DataFrame.from_dict(train_val_summary_dict, orient='columns')
	summarydf['epochs_index'] = np.arange(starting_epoch, ending_epoch)
	summarydf.set_index('epochs_index', drop=True, append=False, inplace=True)
	summary_csv_path = os.path.join('./', 'study{0:02d}_trainval_summary.csv'.format(studyIDX))
	tr.save_append_df(summary_csv_path, summarydf, START)

	## Save Sample Dictionaries
	trainsamplesdf = pd.DataFrame.from_dict(train_samples_dict, orient='columns')
	trainsamples_csv_path = os.path.join('./', 'study{0:02d}_train_samples.csv'.format(studyIDX))
	tr.save_append_df(trainsamples_csv_path, trainsamplesdf, START)
	valsamplesdf = pd.DataFrame.from_dict(val_samples_dict, orient='columns')
	valsamples_csv_path = os.path.join('./', 'study{0:02d}_val_samples.csv'.format(studyIDX))
	tr.save_append_df(valsamples_csv_path, valsamplesdf, START)

	print("Training and Validation results sucessfully written to csv.")

	#############################################
	## Continue if Necessary
	#############################################
	FINISHED_TRAINING = e+1 > total_epochs
	if not FINISHED_TRAINING:
		new_slurm_file = tr.continuation_setup(new_checkpoint_path, studyIDX, last_epoch=e)
		os.system(f'sbatch {new_slurm_file}')

	#############################################
	## Run Test Set When Training is Complete
	#############################################
	if FINISHED_TRAINING:
		print("Testing Model . . .")
		test_dataloader = tr.make_dataloader(test_dataset, batch_size=batch_size)
		testbatch_ID = 0
		testing_dict = {
						"epoch": [],
						"batch": [],
						"truth": [],
						"prediction": [],
						"loss": []
						}

		with torch.no_grad():
			for testdata in test_dataloader:
				testbatch_ID += 1
				truth, pred, loss = tr.eval_datastep(testdata, 
														model,
														loss_fn,
														device)
				testing_dict = tr.append_to_dict(testing_dict, testbatch_ID, truth, pred, loss)


		## Save Testing Info
		del testing_dict["epoch"]
		testingdf = pd.DataFrame.from_dict(testing_dict, orient='columns')
		testingdf.to_csv(os.path.join('./', 'study{0:02d}_testset_results.csv'.format(studyIDX)))
		print('Model testing results saved.')

		## Print Final Summary Statistics
		print('\n')
		print('Model Summary')
		print('=========================================')
		summary = pd.read_csv(os.path.join('./', 'study{0:02d}_trainval_summary.csv'.format(studyIDX)))
		final_train_loss = summary['train_loss'].iloc[-1]
		final_val_loss = summary['val_loss'].iloc[-1]
		print('The final training loss for the model at epoch', e, 'is', round(final_train_loss, 4))
		print('The final valiation loss for the model at epoch', e, 'is', round(final_val_loss, 4))
		print('\n')

		best_val_loss = summary['val_loss'].min()
		best_val_epoch = summary['val_loss'].argmin()
		print('The best validation loss for the model was at epoch', best_val_epoch, 'with a loss of', round(best_val_loss, 4))
		print('\n')

		avg_time = summary['epoch_time'].mean()
		print('Each epoch took on average', round(avg_time, 4), 'minutes to train')
		print('\n')

		avg_test_loss = testingdf['loss'].mean()
		print('The average loss over the test set is:', round(avg_test_loss, 4))

		print('\n')
		print('STUDY {0:02d} COMPLETE'.format(studyIDX))