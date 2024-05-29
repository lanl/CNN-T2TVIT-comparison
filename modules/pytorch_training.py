# PYTORCH MODEL TRAINING FUNCTIONS
"""
Contains functions for training, validating, and testing a pytorch model.
"""
####################################
## Packages
####################################
import os
import sys
import glob
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

####################################
## Make Dataloader form DataSet
####################################
def make_dataloader(dataset: torch.utils.data.Dataset,
						batch_size: int=8,
						num_batches: int=100):
		""" Function to create a pytorch dataloader from a pytorch dataset
				https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
			Each dataloader has batch_size*num_batches samples randomly selected from the dataset

			num_workers: behavior training on dodona
				=0 if not specified, data is loaded in the main process;
					trains slower if multiple models being trained on the same node
				=1 seperates the data from the main process;
					training speed unaffected by multiple models being trained
				=2 splits data across 2 processors;
					cuts training time in half from num_workers=1
				>2 diminishing returns on training time

			persistant_workers:
				training time seems minimally affected, slight improvement when =True

			Args:
				dataset(torch.utils.data.Dataset): dataset to sample from for data loader
				batch_size (int): batch size
				num_batches (int): number of batches to include in data loader

			Returns:
				dataloader (torch.utils.data.DataLoader): pytorch dataloader made from calico model dataset
		"""
		randomsampler = RandomSampler(dataset, num_samples=batch_size*num_batches)
		dataloader = DataLoader(dataset, batch_size=batch_size, sampler=randomsampler, num_workers=4, persistent_workers=True)

		return dataloader

####################################
## Saving Results
####################################
def save_append_df(path: str, df: pd.DataFrame, START: bool):
	""" Function to save/append dataframe contents to a csv file

		Args:
			path (str): path of csv file
			df (pd.DataFrame): pandas dataframe to save
			START (bool): indicates if the file path needs to be initiated

		Returns:
			No Return Objects
	"""
	if START:
		assert not os.path.isfile(path), 'If starting training, '+path+' should not exist.'
		df.to_csv(path, header=True, index = True, mode='x')
	else:
		assert os.path.isfile(path), 'If continuing training, '+path+' should exist.'
		df.to_csv(path, header=False, index = True, mode='a')

def append_to_dict(dictt: dict, batch_ID: int, truth, pred, loss):
	""" Function to appending sample information to a dictionary
		Dictionary must be initialized with correct keys

		Args:
			dictt (dict): dictionary to append sample information to
			batch_ID (int): batch ID number for samples
			truth (): array of truth values for batch of samples
			pred (): array of prediction values for batch of samples
			loss (): array of loss values for batch of samples

		Returns:
			dictt (dict): dictionary with appended sample information
	"""
	batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]
	for i in range(batchsize):
		dictt["epoch"].append(0) # To be easily identified later
		dictt["batch"].append(batch_ID)
		dictt["truth"].append(truth.cpu().detach().numpy().flatten()[i])
		dictt["prediction"].append(pred.cpu().detach().numpy().flatten()[i])
		dictt["loss"].append(loss.cpu().detach().numpy().flatten()[i])

	return dictt

####################################
## Continue Slurm Study
####################################
def continuation_setup(checkpointpath, studyIDX, last_epoch):
    """ Function to generate the training.input and training.slurm files for continuation of model training
        Args:
            checkpointpath (str): path to model checkpoint to load in model from
            studyIDX (int): study ID to include in file name
            last_epoch (int): numer of epochs completed at this checkpoint
        Returns:
            new_training_slurm_filepath (str): name of slurm file to submit job for continued training
    """
    ## Identify Template Files
    training_input_tmpl = "./training_input.tmpl"
    training_slurm_tmpl = "./training_slurm.tmpl"

    ## Make new training.input file
    with open(training_input_tmpl, 'r') as f:
        training_input_data = f.read()
    new_training_input_data = training_input_data.replace('<CHECKPOINT>', checkpointpath)        
    new_training_input_filepath = 'study{0:02d}_restart_training_epoch{1:03d}.input'.format(studyIDX, last_epoch+1)
    with open(os.path.join('./', new_training_input_filepath), 'w') as f:
        f.write(new_training_input_data)

    with open(training_slurm_tmpl, 'r') as f:
        training_slurm_data = f.read()
    new_training_slurm_filepath = 'study{0:02d}_restart_training_epoch{1:03d}.slurm'.format(studyIDX, last_epoch+1)
    new_training_slurm_data = training_slurm_data.replace('<INPUTFILE>', new_training_input_filepath)
    new_training_slurm_data = new_training_slurm_data.replace('<epochIDX>', '{0:03d}'.format(last_epoch+1))         
    with open(os.path.join('./', new_training_slurm_filepath), 'w') as f:
        f.write(new_training_slurm_data)

    return new_training_slurm_filepath

####################################
## Training on a Datastep
####################################
def train_datastep(data: tuple, 
					model,
					optimizer,
					loss_fn,
					device: torch.device):
	""" Function to complete a training step on a single sample

		Args:
			data (tuple): tuple of model input and corresponding ground truth
			model (loaded pytorch model): model to train
			optimizer (torch.optim): optimizer for training set
			loss_fn (torch.nn Loss Function): loss function for training set
			device (torch.device): device index to select

		Returns:
			loss (): evaluated loss for the data sample
	"""
	## Set model to train
	model.train()

	## Extract data
	(inpt, truth) = data
	inpt = inpt.to(device)
	truth = truth.to(torch.float32).unsqueeze(-1).to(device)
	
	## Perform a forward pass
	pred = model(inpt)
	loss = loss_fn(pred, truth)

	## Perform backpropagation and update the weights
	optimizer.zero_grad()
	loss.mean().backward()
	optimizer.step()

	return truth, pred, loss

####################################
## Evaluating on a Datastep
####################################
def eval_datastep(data: tuple, 
					model,
					loss_fn,
					device: torch.device):
	""" Function to complete a validation step on a single sample

		Args:
			data (tuple): tuple of model input and corresponding ground truth
			model (loaded pytorch model): model evaluate
			loss_fn (torch.nn Loss Function): loss function for training set
			device (torch.device): device index to select

		Returns:
			loss (): evaluated loss for the data sample
	"""
	## Set model to eval
	model.eval()

	## Extract data
	(inpt, truth) = data
	inpt = inpt.to(device)
	truth = truth.to(torch.float32).unsqueeze(-1).to(device)

	## Perform a forward pass
	pred = model(inpt)
	loss = loss_fn(pred, truth)

	return truth, pred, loss

######################################
## Training & Validation for an Epoch
######################################
def train_epoch(training_data,
				validation_data, 
				model,
				optimizer,
				loss_fn,
				summary_dict: dict,
				train_sample_dict: dict,
				val_sample_dict: dict,
				device: torch.device):
	""" Function to complete a training step on a single sample

		Args:
			training_data (torch.dataloader): dataloader containing the training samples
			validation_data (torch.dataloader): dataloader containing the validation samples
			model (loaded pytorch model): model to train
			optimizer (torch.optim): optimizer for training set
			loss_fn (torch.nn Loss Function): loss function for training set
			summary_dict (dict): dictionary to save epoch stats to
			train_sample_dict (dict): dictionary to save training sample stats to
			val_sample_dict (dict): dictionary to save validation sample stats to
			device (torch.device): device index to select

		Returns:
			summary_dict (dict): dictionary with epoch stats
			train_sample_dict (dict): dictionary with training sample stats
			val_sample_dict (dict): dictionary with validation sample stats
	"""
	## Initialize things to save
	startTime = time.time()
	trainbatches = len(training_data)
	valbatches = len(validation_data)
	trainbatch_ID = 0
	valbatch_ID = 0

	## Train on all training samples
	for traindata in training_data:
		trainbatch_ID += 1
		truth, pred, train_loss = train_datastep(traindata, 
													model,
													optimizer,
													loss_fn,
													device)
		train_sample_dict = append_to_dict(train_sample_dict, trainbatch_ID, truth, pred, train_loss)
	train_batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

	## Evaluate on all validation samples
	with torch.no_grad():
		for valdata in validation_data:
			valbatch_ID += 1
			truth, pred, val_loss = eval_datastep(valdata, 
													model,
													loss_fn,
													device)
			val_sample_dict = append_to_dict(val_sample_dict, valbatch_ID, truth, pred, val_loss)
	val_batchsize = np.shape(truth.cpu().detach().numpy().flatten())[0]

	## Calcuate the Epoch Average Loss
	train_samples = train_batchsize * trainbatches
	val_samples = val_batchsize * valbatches
	avgTrainLoss = np.sum(train_sample_dict["loss"][-train_samples:]) / train_samples
	avgValLoss = np.sum(val_sample_dict["loss"][-val_samples:]) / val_samples
	summary_dict["train_loss"].append(avgTrainLoss)
	summary_dict["val_loss"].append(avgValLoss)

	## Calculate Time
	endTime = time.time()
	epoch_time = (endTime - startTime) / 60
	summary_dict["epoch_time"].append(epoch_time)

	return summary_dict, train_sample_dict, val_sample_dict