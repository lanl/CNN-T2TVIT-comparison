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

sys.path.insert(0, os.path.abspath('../modules'))
from pytorch_dataset import PVI_SingleField_DataSet
import pytorch_training as tr

#############################################
## Inputs
#############################################
descr_str = 'Python Script to find which saved model checkpoint has the best test set accuracy'
parser = argparse.ArgumentParser(prog='Find Best Model Checkpoint',
								 description=descr_str,
								 fromfile_prefix_chars='@')

parser.add_argument('-SID', '--study_ID',
					action='store',
					type=str,
					default='.',
					help='ID number of selected study')

parser.add_argument('-D', '--directory',
					action='store',
					type=str,
					default='.',
					help='Path to directory containing model folders')

parser.add_argument('-M', '--model',
					action='store',
					type=str,
					default='CNN',
					choices=['CNN', 'VIT', 'T2TVIT'],
					help='Architecture type of the study')

#############################################
#############################################
if __name__ == '__main__':

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args = parser.parse_args()
	SID = args.study_ID
	directory = args.directory
	model = args.model

	## Initiate Model
	if model=='CNN':
		from CNN_modules import CNN
		dir_path = os.path.join(directory, 'CNN_models')
		hyperparameters = pd.read_csv(os.path.join(dir_path, 'hyperparameters.csv'), index_col=0)
		depth = hyperparameters[hyperparameters.index==int(SID)]['DEPTH'].item()
		model = CNN(img_size = (1, 1700, 500),
					size_threshold = (8, 8),
					kernel = 5,
					features = 12, 
					interp_depth = depth,
					act_layer = nn.GELU,
					hidden_features = 20)

	elif model=='VIT':
		from ViT_modules import ViT_Model
		dir_path = os.path.join(directory, 'ViT_Models')
		model = ViT_Model(img_size = (1, 1700, 500),
						patch_size = 50,
						token_len = 1024,
						preds = 1,
						num_heads = 4,
						Encoding_hidden_chan_mul = 1.0,
						depth = 12,
						qkv_bias = False,
						qk_scale = None,
						act_layer = nn.GELU,
						norm_layer = nn.LayerNorm)

	elif model=='T2TVIT':
		from ViT_modules import T2T_ViT
		dir_path = os.path.join(directory, 'T2T_ViT_Models')
		model = T2T_ViT(img_size = (1, 1700, 500),
						softsplit_kernels = (31, 3, 3),
						preds = 1,
						token_len = 1024,
						token_chan = 64,
						num_heads = 4,
						T2T_hidden_chan_mul = 1.0,
						Encoding_hidden_chan_mul = 1.0,
						depth = 12,
						qkv_bias = False,
						qk_scale = None,
						act_layer = nn.GELU,
						norm_layer = nn.LayerNorm)

	## Get Summary
	summary_csv_name = 'study'+SID+'_trainval_summary.csv'
	summary_csv_path = os.path.join(dir_path, 'study_'+SID, summary_csv_name)
	summary_csv = pd.read_csv(summary_csv_path, index_col=0)

	## Get Checkpoint Places
	max_epoch = summary_csv.index.max()
	checkpoints = np.arange(0, max_epoch+1, 50)[1:]
	best_checkpoint = summary_csv.loc[checkpoints]['val_loss'].idxmin()

	## Load Checkpoint
	model.to(device)
	checkpoint_file = 'study'+SID+'_modelState_epoch{:03d}.pth'.format(best_checkpoint)
	checkpoint_path = os.path.join(dir_path, 'study_'+SID, checkpoint_file)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["modelState"])

	## Actually Testing
	loss_fn = nn.MSELoss(reduction='none')
	test_dataset = PVI_SingleField_DataSet(filelist = os.path.join(directory, 'filelists', 'PVI_only_test_samples.txt'),
											input_field = 'rho',
											predicted = 'ptw_scale',
											design_file = os.path.join(directory, 'design_nc231213_Sn_MASTER.csv'))
	test_dataloader = tr.make_dataloader(test_dataset, batch_size=4)
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
	testingdf.to_csv(os.path.join(dir_path, 'study_'+SID, 'study'+SID+'_bestCHKPT_testset_results.csv'))

	avg_test_loss = testingdf['loss'].mean()
	print('The best model checkpoint was epoch', best_checkpoint, 'with a validation loss of', summary_csv.loc[best_checkpoint, 'val_loss'])
	print('The average loss over the test set is:', avg_test_loss)
	print('Model testing results saved.')