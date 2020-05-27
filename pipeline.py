import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision 
from torchvision import transforms

import numpy as np 
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

from helper import *
from ocr_test import *

import cv2

modelPATH = 'D:/Deep Learning Trained Models/Forms/100.pt'
imgPATH = "D:/Deep Learning Testing Data/DRex/Original Forms/5.jpg"
#imgPATH = "D:/Deep Learning Testing Data/DRex/Testing Images/SG9AL.jpg"

print("==============================================================")
print("Initiating Trained Model...")
model = get_pretrained_model(modelPATH)
model.eval()

def single_pred_pass(original_image, shadow):
	print("\nPre-Processing Image for Prediction...")
	prediction_image = get_prediction_image(original_image, shadow)

	print("Making Prediction and Segmenting Image to get Boxes...")
	boxes = custom_predict(model, pil(original_image), pil(prediction_image), False, True)

	return boxes

def single_box_pass(original_image, first_pass):
	boxes = single_pred_pass(original_image, False)
	if len(boxes) == 1 and first_pass:
		print("ACTIVATED SHADOW REMOVAL")
		boxes = single_pred_pass(original_image, True)

	return boxes


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
		#pil(image).save("./Condition/"+str(i)+".png")
		#image = cv2.imread("./Condition/"+str(i)+".png")
		#temp_boxes = single_box_pass(image, False)
		temp_boxes = custom_predict(model, pil(image), pil(isolate_printed_text(pil(image))), True, True)

		if validate_block(image, temp_boxes):
			print("Segmenting again...")
			#plt.imshow(pil(org_img))
			#plt.show()
			#image = get_prediction_image(org_img, False)
			temp = custom_predict(model, pil(org_img), pil(org_img), True, True)
			print("New Number of Boxes for "+str(i)+": "+str(len(temp)))
			for box in temp:
				data.append(box)
		else:
			data.append(pil(org_img))

	for i in range(len(data)):
		data[i].save("./Condition/"+str(i)+".png")
	print("\nLength of Entire Sequence: "+str(len(data)))

	print("Using OCR and writing Labels and Info...\n")
	for i in range(len(data)):
		PATH = "./Condition/"+str(i)+".png"
		
		label = get_text(PATH, data[i], True)
		info = get_text(PATH, data[i], False)

		if info == ' ' or label == ' ':
			continue
		else:
			label, info = clean_text(label, info)

			with open("./Labels and Content/labels.txt", 'a', encoding="utf-8") as f:
				f.write(str(label))
				f.write('\n')

			with open("./Labels and Content/content.txt", 'a', encoding="utf-8") as f:
				f.write(str(info))
				f.write('\n')

	'''i = 5
	image = cv2.imread("./Temp/"+str(i)+".png")
	plt.imshow(pil(image))
	plt.show()

	image = isolate_printed_text(image)
	pil(image).save("./Condition/"+str(i)+".png")
	image = cv2.imread("./Condition/"+str(i)+".png")
	temp_boxes = single_box_pass(image, False)
	if validate_block(image, temp_boxes):
		print(len(temp_boxes))
		print("Yes")
			
	for i in range(len(temp_boxes)):
		plt.imshow(temp_boxes[i])
		plt.axis('off')
		plt.show()

	for i in range(len(boxes)):
		print(i)
		image = cv2.imread("./Temp/"+str(i)+".png")
		temp_boxes = single_box_pass(image, False)
		if len(temp_boxes) > 3:
			for j in range(len(temp_boxes)):
				data.append(temp_boxes[j])
		else:
			data.append(pil(image))

	print(len(data))

	delete_files_from_folder("./Temp")

	for i in range(len(data)):
		data[i].save("./Temp/"+str(i)+".png") '''


	'''if flag:
					#delete_files_from_folder("C:/Users/Rohan/Desktop/College Work/BTECH FINAL PROJECT/Code Attempt 2/Temp")
					#print("Files Deleted")'''
	


	
pipeline(imgPATH, False)


'''
print("Using OCR and writing Labels and Info...\n")
	for i in range(len(boxes)):
		PATH = "./Temp/"+str(i)+".png"
		
		label = get_text(PATH, boxes[i], True)
		info = get_text(PATH, boxes[i], False)

		if info == ' ' or label == ' ':
			continue
		else:
			label, info = clean_text(label, info)

			with open("./Labels and Content/labels.txt", 'a', encoding="utf-8") as f:
				f.write(str(label))
				f.write('\n')

			with open("./Labels and Content/content.txt", 'a', encoding="utf-8") as f:
				f.write(str(info))
				f.write('\n')'''


'''
0, 2, 7, 8, 9, 10, 11, 
if block has more than 2 new blocks, re-segment, else, leave alone


dont even bother performing 2 level segmentation on a pic if it only has one recognized box initially
refer to 
'''