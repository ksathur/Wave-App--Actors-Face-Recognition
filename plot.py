from __future__ import print_function, division
import torch 
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import time
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import sys

#################################################
	# Confusion matrix
#################################################
def plot_confusion_matrix(confusion_matrix, file_name, normalize =  False, title = None, cmap = plt.cm.Blues):
	classes = []
	for i in range(confusion_matrix.shape[0]):
		classes.append(i)
		
	if normalize:
		confusion_matrix = confusion_matrix.astype(float)
		for i in range(confusion_matrix.shape[0]):
			confusion_matrix[i] = confusion_matrix[i]/np.sum(confusion_matrix[i], dtype = float) if (sum(confusion_matrix[i])!= 0) else 0 

	fig, ax = plt.subplots()
	im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=np.arange(confusion_matrix.shape[1]), yticks=np.arange(confusion_matrix.shape[0]),
		xticklabels=classes, yticklabels=classes,
		title=title,
		ylabel='True label',
		xlabel='Predicted label')

	fmt = '.2f' #if normalize else 'd'
	thresh = confusion_matrix.max() / 2.
	for i in range(confusion_matrix.shape[0]):
		for j in range(confusion_matrix.shape[1]):
			ax.text(j, i, format(confusion_matrix[i, j], fmt), ha="center", va="center", 
				color="white" if confusion_matrix[i, j] > thresh else "black")


	fig.tight_layout()
	plt.savefig(file_name)
	# plt.show()
	return ax

################################################
	# Plotting Confusion Matrix
################################################

acc_file_name = 'dumps/accuracy/face_detection.npz'

arr = np.load(acc_file_name)

number_of_correct_by_class = arr['arr_0']
number_by_class = arr['arr_1']
cfm = arr['arr_2']


cfm_name = 'dumps/accuracy/cfm.png'

plot_confusion_matrix(cfm, cfm_name, normalize =  True, title = None, cmap = plt.cm.Blues)













































