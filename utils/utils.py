from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################################################
	# Reading images
################################################
def read_load_inputs_labels(image_names, labels, transform = None):
	img_inputs = []
	for i in range(len(image_names)):
		image = io.imread(image_names[i])
		if transform:
			image = transform(Image.fromarray(image))
		img_inputs.append(image)
	inputs = torch.stack(img_inputs)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	inputs = inputs.to(device)
	labels = labels.to(device)
	return inputs, labels

################################################
	# Training model
################################################
def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, transform = None):
	model.train()
	for epoch in range(1, num_epochs + 1):
		print('Epoch : ', epoch)
		for image_names, labels in train_loader:
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				outputs = model(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		scheduler.step()
	return model

################################################
	# Testing model
################################################
def test_model(model, criterion, optimizer, test_loader, test_loader_size, number_of_classes, model_file_name, transform = None):
	model.eval()
	test_loss = 0.0
	corrects = 0
	number_of_correct_by_class = np.zeros((1, number_of_classes), dtype = int)
	number_by_class = np.zeros((1, number_of_classes), dtype = int)
	confusion_matrix = np.zeros((number_of_classes, number_of_classes), dtype = int)
	for image_names, labels in test_loader:
		with torch.no_grad():
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			loss = criterion(outputs, labels)
			test_loss += loss.item() * inputs.size(0)
			corrects += torch.sum(preds == labels.data)
			for k in range(preds.size(0)):
				confusion_matrix[labels[k]][preds[k]] = confusion_matrix[labels[k]][preds[k]] + 1
				number_by_class[0][labels[k]] = number_by_class[0][labels[k]] + 1
				if preds[k] == labels[k]:
					number_of_correct_by_class[0][preds[k]] = number_of_correct_by_class[0][preds[k]] + 1
	
	print(number_of_correct_by_class)
	print(number_by_class)
	epoch_loss = test_loss / test_loader_size
	print(corrects.item())
	print(test_loader_size)
	epoch_acc = corrects.double() / test_loader_size
	print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

	torch.save(model.state_dict(), model_file_name)
	return	number_of_correct_by_class, number_by_class, confusion_matrix

################################################
	# Sample tester
################################################
def sample_tester(model, test_loader, transform = None):
	model.eval()
	label_name_list = ['Audi A8 2020 Car', 'Honda Civic 2011 Car', 'Honda Fit GP5 2015 Car', 'Honda Vezel 2015 SUV', 'landcuiser predo 2018', 'Nissan X -Trail Hybrid 2019 SUV', 'Suzuki Wragon R Stingray 2018 Car', 'Suzuzki Swift RS 2019 Car', 'Toyota Aqua 2019 Car', 'Toyota Axio 2015 Car', 'Toyota Vitz 2019 Car']
	for image_names, labels in test_loader:
		with torch.no_grad():
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			for k in range(preds.size(0)):
				print(image_names[k], label_name_list[preds[k]])
	