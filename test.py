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

original_image = cv2.imread("< PATH TO FORM IMAGE >")

print("Pre-Processing Image for Prediction...")
prediction_image = get_prediction_image(original_image)

print("Initiating Trained Model...")
model = get_pretrained_model('< PATH TO 100.PT FILE >')
model.eval()

print("Making Prediction and Segmenting Image to get Boxes...")
boxes = custom_predict(model, pil(original_image), pil(prediction_image), False, True)

print("Using OCR and writing Labels and Info...")
for i in range(len(boxes)):
	#CREATE A TEMPORARY FOLDER IN YOUR WORKING DIRECTORY CALLED "TEMP"
	PATH = "./Temp/"+str(i)+".png"
	label = get_cleaned_text(PATH, boxes[i], True)
	info = get_cleaned_text(PATH, boxes[i], False)

	info = info.replace(label, '')

	if info == ' ' or label == ' ':
		continue
	else:
		with open("< PATH TO TXT FILE TO STORE LABELS >", 'a', encoding="utf-8") as f:
			f.write(str(label))
			f.write('\n')

		with open("< PATH TO TXT FILE TO STORE CONTENT >", 'a', encoding="utf-8") as f:
			f.write(str(info))
			f.write('\n')
