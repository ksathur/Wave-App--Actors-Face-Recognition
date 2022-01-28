#################################################
	# csv file writer for face dataset
	# writing order: image_path, label
#################################################
import os
import csv

images_dir = '../dataset/raw/'
class_list = os.listdir(images_dir)

#################################################
	# writing dataset as csv file
#################################################
with open('../dataset/image_path_and_labels.csv', 'w') as train_csvfile:

	filewriter = csv.writer(train_csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	filewriter.writerow(['image_path', 'label'])
	for label, face_class in enumerate(class_list):
		img_name_list = os.listdir(os.path.join(images_dir, face_class))
		img_name_list.sort()
		for img_name in img_name_list:
			img_dir = 'dataset/raw/' + face_class +  '/' + img_name
			filewriter.writerow([img_dir, label])
