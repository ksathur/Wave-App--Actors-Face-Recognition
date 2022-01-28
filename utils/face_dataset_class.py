from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

#################################################
	# Face dataset class
#################################################
class Face(Dataset):
	def __init__(self, csv_file):
		self.image_path_and_labels = np.array(pd.read_csv(csv_file))

	def get_by_class(self, class_id):
		sample = []
		min_no_of_images_in_class = 114
		rough_start_index = min_no_of_images_in_class * class_id
		for i in range(rough_start_index, len(self.image_path_and_labels)): 
			if self.image_path_and_labels[i][1] == class_id:
				image_name = self.image_path_and_labels[i][0]
				label = class_id
				sample.append([image_name, label])
			elif self.image_path_and_labels[i][1] == class_id + 1:
				break
		return sample