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

PATH = "C:/Users/Rohan/Desktop/nsGIo.jpg"
#PATH = "D:/Deep Learning Training Data/Forms Dataset/images/images/3.jpg"
#PATH = "C:/Users/Rohan/Desktop/College Work/BTECH FINAL PROJECT/Associated Documents/Associated Images/RESULTS AND EXAMPLES/ZOOM EXAMPLE/wRlzI.png"
#PATH = "C:/Users/Rohan/Desktop/College Work/BTECH FINAL PROJECT/Code Attempt 2/Temp/9.png"
original_image = cv2.imread(PATH)

print("\nPre-Processing Image for Prediction... \n")

def initiate_pipeline(PATH, remove_shadow):
	prediction_image = get_prediction_image(original_image, remove_shadow)

	print("Initiating Trained Model... \n")
	model = get_pretrained_model('D:/Deep Learning Trained Models/Forms/100.pt')
	model.eval()

	print("Making Prediction and Segmenting Image to get Boxes... \n")
	boxes = custom_predict(model, pil(original_image), pil(prediction_image), False, True)

	return boxes

boxes = initiate_pipeline(PATH, False)
if len(boxes)<3:
	print("Activated Shadow Removal")
	boxes = initiate_pipeline(PATH, True)


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
			f.write('\n')
