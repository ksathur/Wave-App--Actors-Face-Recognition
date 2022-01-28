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

################################################
	# Importing functions from other directory
################################################
sys.path.append('utils')
from face_dataset_class import *
from utils import *

################################################
	# Training parameters
################################################
batch_size = 4
step_size = 7
gamma = 1
number_of_classes = 4
learning_rate = 0.0001
number_train_samples = 90
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
csv_file = 'dataset/image_path_and_labels.csv'
face_dataset = Face(csv_file)

test = []
for class_id in range(number_of_classes):
	data = face_dataset.get_by_class(class_id)
	test += data[number_train_samples : ]

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)
test_loader_size = len(test)

################################################
	# Training and Testing
################################################
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, number_of_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(model_file_name))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

number_of_correct_by_class, number_by_class, confusion_matrix = test_model(model, criterion, optimizer, test_loader, test_loader_size, number_of_classes, model_file_name, transform = transform)


	
