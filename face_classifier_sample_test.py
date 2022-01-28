from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import copy
import pandas as pd
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys
import os

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
	# Sample tester
################################################
def sample_tester(model, test_loader, number_of_classes, transform = None):
	model.eval()
	correct_by_class = np.zeros((number_of_classes), dtype = int)
	label_name_list = ['Person1', 'Person2', 'Person3', 'Person4']
	for image_names, labels in test_loader:
		with torch.no_grad():
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			for k in range(preds.size(0)):
				print(image_names[k], label_name_list[preds[k]])
				correct_by_class[preds[k]] = correct_by_class[preds[k]] + 1
	print('\n')			
	print('############# OUTPUT #############')
	for idx, person in enumerate(label_name_list):
		if correct_by_class[idx] == 0:
			print(person, 'is absent')
			# print(person, 0, 'None')
		elif correct_by_class[idx]%2 == 1:
			print(person, 'is present and he is in the class now')
			# print(person, 1, 'In')
		else:
			print(person, 'is present and he is out of the class now')
			# print(person, 1, 'Out')

################################################
	# Model name loading parameters
################################################
batch_size = 4
model_file_name = 'dumps/model/face_detection.pth'

################################################
	#  Transformations
################################################
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([transforms.Resize(256),
								transforms.CenterCrop(224),
								transforms.ToTensor(),
								transforms.Normalize(mean = mean, std = std),])

################################################
	# Data loading
################################################
test_sample_dir = 'test_sample'
img_name_list = os.listdir(test_sample_dir)
img_name_list.sort()

test = []
for img_name in img_name_list:
	img_dir = 'test_sample/' + img_name
	label = -1
	test.append([img_dir, label])

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

################################################
	# Testing
################################################
number_of_classes = 4
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, number_of_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(model_file_name))

sample_tester(model, test_loader, number_of_classes, transform = transform)

