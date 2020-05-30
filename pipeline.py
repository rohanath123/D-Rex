import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision 
from torchvision import transforms

import numpy as np 
from PIL import Image

from hc import *

import matplotlib
import matplotlib.pyplot as plt

from helper import *
from ocr_test import *

import cv2

modelPATH = 'D:/Deep Learning Trained Models/Forms/100.pt'
#imgPATH = "D:/Deep Learning Testing Data/DRex/New Forms/4.jpg"
imgPATH = "D:/Deep Learning Testing Data/DRex/Original Forms/5.jpg"

model = get_pretrained_model(modelPATH)
model.eval()

def single_pred_pass(original_image, shadow):
	prediction_image = get_prediction_image(original_image, shadow)

	boxes = custom_predict(model, pil(original_image), pil(prediction_image), False, True)

	return boxes

def single_box_pass(original_image, first_pass):
	boxes = single_pred_pass(original_image, False)
	if len(boxes) == 1 and first_pass:
		boxes = single_pred_pass(original_image, True)

	return boxes
'''

def pipeline(imgPATH, flag):
	delete_files_from_folder("./Temp")
	delete_files_from_folder("./Condition")
	delete_files_from_folder("./Labels and Content")

	data = []

	image = cv2.imread(imgPATH)
	boxes = single_box_pass(image, True)

	for i in range(len(boxes)):
		boxes[i].save("./Temp/"+str(i)+".png")

	for i in range(len(boxes)):
		org_img = cv2.imread("./Temp/"+str(i)+".png")

		image = isolate_printed_text(org_img)
		temp_boxes = custom_predict(model, pil(image), pil(image), True, True)

		if validate_block(image, temp_boxes):
			temp = custom_predict(model, pil(org_img), pil(org_img), True, True)
			for box in temp:
				data.append(box)
		else:
			data.append(pil(org_img))

	for i in range(len(data)):
		data[i].save("./Condition/"+str(i)+".png")

	#labels = []
	infos = []
	images = []
	labels = hc()

	for i in range(len(data)):
		PATH = "./Condition/"+str(i)+".png"
		image = cv2.imread(PATH)
		images.append(image)
		
		#label = get_raw_text(PATH, image, True)
		info = get_raw_text(PATH, image, False)

		#labels.append(label)
		infos.append(info)

	labels, infos = process_text(labels, infos, images)

	return labels, infos
'''


def pipeline(imgPATH, flag):
	delete_files_from_folder("./Temp")
	delete_files_from_folder("./Condition")
	delete_files_from_folder("./Labels and Content")

	

	image = cv2.imread(imgPATH)
	boxes = single_box_pass(image, True)

	data = resegment(boxes, model)
	#data = resegment(boxes, model)

	for i in range(len(data)):
		data[i].save("./Condition/"+str(i)+".png")

	#labels = []
	infos = []
	images = []
	labels = hc()

	for i in range(len(data)):
		PATH = "./Condition/"+str(i)+".png"
		image = cv2.imread(PATH)
		images.append(image)
		
		#label = get_raw_text(PATH, image, True)
		info = get_raw_text(PATH, image, False)

		#labels.append(label)
		infos.append(info)

	print(infos)
	infos = process_text(labels, infos, images)
	labelss, infoss = clean_text_data(labels, infos)

	return labelss, infoss


#USE THIS TO CALL PIPELINE WHERE IMGPATH IS PATH TO INPUT FORM
labels, infos = pipeline(imgPATH, False)

for i in range(len(labels)):
	print(labels[i]+': '+infos[i])
