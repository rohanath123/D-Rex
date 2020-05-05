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
#"D:/Deep Learning Training Data/Empty Forms/Empty Forms/13149651.png"
#"C:/Users/Rohan/Desktop/Misc Images"

original_image = cv2.imread("D:/Deep Learning Training Data/Empty Forms/Empty Forms/0030041455.png")

prediction_image = get_prediction_image(original_image)

model = get_pretrained_model('D:/Deep Learning Trained Models/Forms/100.pt')
model.eval()
boxes = custom_predict(model, pil(original_image), pil(prediction_image), False, True)

for i in range(len(boxes)):
	PATH = "C:/Users/Rohan/Desktop/Misc Images/"+str(i)+".png"
	boxes[i].save(PATH)
	text = detect_document(PATH)
	if text == ' ':
		continue
	else:
		print(text)