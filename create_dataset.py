#FILE FOR SAVING ITERATING OVER HANDWRITING DATASET AND SAVING FORMS CREATED FROM distributed_writing.py

import os 

import torch
import torchvision
import torchvision.datasets as dsets 
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt

import random

from distributed_writing import *

pil = transforms.ToPILImage()
tens = transforms.ToTensor()

PATH = 'D:/Deep Learning Training Data/sentences/'
image_set = []
for filename in os.listdir(PATH):
	image_set.append(dsets.ImageFolder(PATH+filename, transform = transforms.ToTensor()))
	#images = images + [image_set[i][0] for i in range(len(image_set))]

num = 0
i = 0
for iset in image_set:
	print(i+1)
	i = i+1

	images = [iset[j][0] for j in range(len(iset))] 
	print(len(images))
	make_and_save_forms(images, int(len(images)/4), 'D:/Deep Learning Training Data/Forms Created Dataset/')
	del images
	'''forms = make_forms(images, int(len(images)/4))
				num = save_forms_set(forms, 'D:/Deep Learning Training Data/Forms Created Dataset/', num)
				del images
				del forms
				i = i+1
				if i%2 == 0:
					break
			'''

'''
#images = [images[i][0] for i in range(len(images))]
#print(len(images))

forms = make_forms(images, int(len(images)/4))

print(len(forms))

save_forms_set(forms, 'D:/Deep Learning Training Data/Forms Created Dataset/')'''