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

original_image = cv2.imread("D:/Deep Learning Training Data/Forms Dataset/images/images/5.jpg")

print("Pre-Processing Image for Prediction...")
prediction_image = get_prediction_image(original_image)

print("Initiating Trained Model...")
model = get_pretrained_model('D:/Deep Learning Trained Models/Forms/100.pt')
model.eval()

print("Making Prediction and Segmenting Image to get Boxes...")
boxes = custom_predict(model, pil(original_image), pil(prediction_image), False, True)

print("Using OCR and writing Labels and Info...")
for i in range(len(boxes)):
	PATH = "./Temp/"+str(i)+".png"
	label = get_cleaned_text(PATH, boxes[i], True)
	info = get_cleaned_text(PATH, boxes[i], False)

	info = info.replace(label, '')

	if info == ' ' or label == ' ':
		continue
	else:
		with open("./Labels and Content/sunfeast_labels.txt", 'a', encoding="utf-8") as f:
			f.write(str(label))
			f.write('\n')

		with open("./Labels and Content/sunfeast_content.txt", 'a', encoding="utf-8") as f:
			f.write(str(info))
			f.write('\n')