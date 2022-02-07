################################################
	# Kanagarajah Sathursan
	# ksathursan1408@gmail.com
################################################
from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms, utils
from skimage import io, transform
from PIL import Image
import os
import time
import os.path
from h2o_wave import main, app, Q, ui, handle_on , on

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
	device = torch.device("cpu")
	inputs = inputs.to(device)
	labels = labels.to(device)
	return inputs, labels

	# Sample tester
################################################
def sample_tester(model, test_loader, number_of_classes, transform = None):
	model.eval()
	label_name_list = ['Aaron Eckhart', 'Adam Brody', 'Bradley Cooper', 'Adrien Brody']
	image_names_list = []
	pred_label_list = []
	for image_names, labels in test_loader:
		with torch.no_grad():
			inputs, labels = read_load_inputs_labels(image_names, labels, transform = transform)
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
			for k in range(preds.size(0)):
				image_names_list.append(image_names[k])
				pred_label_list.append(label_name_list[preds[k]])
	return image_names_list, pred_label_list

	# Model name loading parameters
################################################
batch_size = 4
model_file_name = 'dumps/model/face_detection.pth'

	# Transformations
################################################
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([transforms.Resize(256),
								transforms.CenterCrop(224),
								transforms.ToTensor(),
								transforms.Normalize(mean = mean, std = std),])

	# Grid image card
################################################
def image_card(q,plot_name,x,y,img_path,label):
	q.page[plot_name] = ui.tall_article_preview_card(
            box=f'{x} {y} 2 3',
            name='img_card',
            title=f'{label}',
            image=f'{img_path}',
    )


@on('predict')
async def predict_mtd(q: Q):
	
	# Data loading
################################################
	path_dict = {}
	for path in q.client.img_paths:
		rel_path = path.split("/")[-1]
		path_dict[rel_path] = path
		
	test = []
	for img_name in path_dict.keys():
		img_dir = 'test_sample/' + img_name
		label = -1
		test.append([img_dir, label])

	test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
	
	# Testing
################################################
	number_of_classes = 4
	model = models.resnet18(pretrained=True)
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, number_of_classes)
	device = torch.device("cpu")
	model = model.to(device)
	model.load_state_dict(torch.load(model_file_name, map_location=device))

	image_names_list, pred_label_list = sample_tester(model, test_loader, number_of_classes, transform = transform)

	time.sleep(3)
	grid_coor = [[3,2], [5,2], [7,2], [9,2], [3,5], [5,5], [7,5], [9,5], [3,8], [5,8], [7,8], [9,8]]

	del q.page["example2"]
	for  i,img in enumerate(image_names_list):
		image_card(q, img, grid_coor[i][0], grid_coor[i][1], path_dict[img.split("/")[-1]], pred_label_list[i])



@app('/actors_face_detection')
async def serve(q: Q):
	if not await handle_on(q):
		await dashboard(q)

			
	await q.page.save()

@on('goback')
async def dashboard(q: Q):
	q.page["header"] = ui.header_card(
			box="1 1 10 1",
			title="Actors Face Recognition",
			subtitle="",
			image='https://wave.h2o.ai/img/h2o-logo.svg',
			color='primary'
		)
		
	links = q.args.user_files

	if links:
		del q.page["example"]
		
		items = [ui.text_xl('Files uploaded!')]
		img_path = []

		for link in links:
			local_path = await q.site.download(link, './test_sample')
			img_path.append(link)
			size = os.path.getsize(local_path)
			items.append(ui.link(label=f'{os.path.basename(link)} ({size} bytes)', download=True, path=link))

		q.client.img_paths = img_path
		items.append(ui.button(name='back', label='Back', primary=True))
		q.page['example'].items = items

		q.page['example3'] = ui.form_card(
			box='1 2 2 9', 
			items=[ 
				ui.text_l('Click on the \'Start\' button to start the face recognition. Click on the \'Back\' button to go back and select the images.\n\n  '),
				ui.button(name="predict", label="Start", primary=True),
				ui.button(name="goback", label=" Back ", primary=True)]
		)
		
	else:
		q.page['example'] = ui.form_card(box='1 2 2 9', items=[
			ui.text_l('This is a face recognition system that can recognizes the faces of the following actors:\n 1. Aaron Eckhart\n 2. Adam Brody\n 3. Bradley Cooper\n 4. Adrien Brody\n \nAfter selecting the images of these actors, click on the \'Upload\' button to start the uploading.'),
			ui.file_upload(name='user_files', label='Upload', multiple=True, max_size=12),
		])
	
		q.page['example2'] = ui.tall_article_preview_card(
			box='3 2 8 9', 
			title='',
    		subtitle='',
			image="https://img.freepik.com/free-vector/face-recognition-low-poly-wireframe-banner-template-futuristic-computer-technology-smart-identification-system-poster-polygonal-design-facial-scan-3d-mesh-art-with-connected-dots_201274-4.jpg?size=626&ext=jpg"
		)
