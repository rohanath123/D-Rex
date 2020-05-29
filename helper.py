import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision 
from torchvision import transforms

import numpy as np 
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import pickle

import random
import cv2

import os, shutil

from unet_architecture import *
from ocr_test import *

tens = transforms.ToTensor()
pil = transforms.ToPILImage()
resize = transforms.Resize((512, 512))
gray = transforms.Grayscale(3)

transforms_set = transforms.Compose([transforms.Grayscale(3), transforms.Resize((512, 512)), transforms.ToTensor()])

to_remove = ['-', '"', ':', '.', '(', ')', '--', '—']

def threshold(img, thresh_val):
  img = img >= thresh_val
  img = img.float()
  return img

def anti_thresh(img):
  img = img <= 0.5
  img = img.float()
  return img

import math
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def get_white_sections(img, msk):
  whites_active = False
  white_sections_img = []
  white_sections_msk = []
  single_section_img = []
  single_section_msk = []

  for i in range(len(msk)):
    if 0.0 in msk[i]:
      single_section_img.append(img[i].tolist())
      single_section_msk.append(msk[i].tolist())
      whites_active = True
    else:
      if whites_active:
        white_sections_img.append(single_section_img)
        white_sections_msk.append(single_section_msk)
        single_section_img = []
        single_section_msk = []
        whites_active = False
  if len(white_sections_img) == 0:
        white_sections_img.append(single_section_img)
        white_sections_msk.append(single_section_msk)
  return white_sections_img, white_sections_msk

def add_extra_rows(img):
	img = img.tolist()
	blank = [[255] * len(img)]
	img = blank + img + blank
	return img

def get_boxes(img, msk):
  img_boxes = []
  msk_boxes = []
  img_sections, msk_sections = get_white_sections(img, msk)
  print("Number of Boxes Identified: "+str(len(img_sections)))

  for i in range(len(msk_sections)):
    img_box, msk_box = get_white_sections(np.transpose(img_sections[i]), np.transpose(msk_sections[i]))
    for j in range(len(img_box)):
      img_boxes.append(np.transpose(img_box[j]))
      msk_boxes.append(np.transpose(msk_box[j]))
  return img_boxes, msk_boxes

def custom_predict(model, original_image, prediction_image, threshold_image, threshold_output):
  custom_resize = transforms.Resize((tens(original_image).size()[1], tens(original_image).size()[2]))

  input = tens(gray(resize(prediction_image)))
  if threshold_image:
    input = threshold(input, 0.4)
  
  output = model(input.cuda().view(1, 3, 512, 512))
  output = output.cpu().view(1, 512, 512)

  if threshold_output:
    output = threshold(output, 0.5)
  
  output = tens(custom_resize(pil(output)))

  x, y = get_boxes(tens(original_image)[0], output[0])
  boxes = []

  for box in x:
  	try:
  		boxes.append(pil(torch.FloatTensor([box])))
  	except:
  		print("Empty Block Caught")
  return boxes


def enhance(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.GaussianBlur(image, (5, 5), 0.35)
  ret3,th3 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return th3

def make_3_channels(image):
  temp = torch.cat([image, image])
  temp = torch.cat([temp, image])
  return temp

def remove_lines(image):
  result = image.copy()
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
  remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
  cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      cv2.drawContours(result, [c], -1, (255,255,255), 5)
      
  vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
  remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
  cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      cv2.drawContours(result, [c], -1, (255,255,255), 5)
  return result

def remove_shadow(image):
	rgb_planes = cv2.split(image)

	result_planes = []
	result_norm_planes = []
	for plane in rgb_planes:
	    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
	    bg_img = cv2.medianBlur(dilated_img, 21)
	    diff_img = 255 - cv2.absdiff(plane, bg_img)
	    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
	    result_planes.append(diff_img)
	    result_norm_planes.append(norm_img)

	result = cv2.merge(result_planes)
	result_norm = cv2.merge(result_norm_planes)
	return result_norm

def get_prediction_image(org_image, shadow):	
  image = remove_lines(org_image)
  if shadow:
  	image = remove_shadow(org_image)
  image = enhance(image)
  image = torch.tensor(image)
  image = make_3_channels(image.view(1, image.size()[0], image.size()[1]))
  return image

def get_pretrained_model(PATH):
  model = UNetModel()
  model.load_state_dict(torch.load(PATH))
  model = model.cuda()

  return model

def delete_files_from_folder(PATH):
  folder = PATH
  for filename in os.listdir(folder):
      file_path = os.path.join(folder, filename)
      try:
          if os.path.isfile(file_path) or os.path.islink(file_path):
              os.unlink(file_path)
          elif os.path.isdir(file_path):
              shutil.rmtree(file_path)
      except Exception as e:
          print('Failed to delete %s. Reason: %s' % (file_path, e))

'''def isolate_printed_text(image):
  image = tens(image)
  image = image > 0.2
  image = image.float()
  return image'''

def isolate_printed_text_recognition(image):
  image = image > 35
  image = tens(image.astype(np.float))
  image = image.float()
  return image

def isolate_printed_text(image):
  image = image >= (image.min() + 35)
  image = tens(image.astype(np.float))
  image = image.float()
  return image

def isolate_printed_text2(image):
	image = tens(image)
	image = image != 0
	image = image.float()
	return image

def get_text(PATH, image, isolate):
  if isolate:
    image = isolate_printed_text2(image)
    image = pil(image)

  image.save(PATH)

  text = detect_document(PATH)
  text = ''.join([c for c in text if c not in to_remove])

  if ''.join(set(text)) == ' ':
    return ' '
  else:
    return text

def get_raw_text(PATH, image, isolate):
  if isolate:
    image = isolate_printed_text_recognition(image)
    
  image = pil(image)
  image.save(PATH)

  text = detect_document(PATH)

  return text

def clean_text(labels, infos):
  new_labels = []
  new_infos = []
  for label in labels:
    new_labels.append(''.join([c for c in label if c not in to_remove]))
  for info in infos:
    new_infos.append(''.join([c for c in info if c not in to_remove]))
  return new_labels, new_infos

def remove_words(labels, infos):
  flag = False
  while True:
    for i in range(len(labels)):
      if not all(item in infos[i].split(' ') for item in labels[i].split(' ')):
        break
      if i == len(labels)-1:
        flag = True
    if flag:
      break
    labels.pop(i)
  for i in range(len(labels)):
    for word in labels[i].split(' '):
      infos[i] = infos[i].replace(word, '', 1)
    infos[i] = infos[i].strip()
  return labels, infos

def validate_block(image, boxes):
	percentages = [(np.count_nonzero(tens(boxes[i])[0].numpy() == 0.0)/torch.numel(tens(boxes[i])[0])) for i in range(len(boxes))]
	sizes = [tens(boxes[i]).size()[1] for i in range(len(boxes))]
	valid = [i for i in range(len(boxes)) if sizes[i] >= 35 and sizes[i] <= 150 and percentages[i] >= 0.03]
	if len(valid) > 2:
		return True
	else:
		return False

def resultant_percentage(image):
	image = isolate_printed_text(remove_lines(image))
	perc = (np.count_nonzero(image[0] == 0.0)/torch.numel(image[0]))
	return perc

def classify_block(image):
	perc = resultant_percentage(image)
	if perc < 0.0047:
		return "Handwriting"
	elif perc > 0.07:
		return "Label"
	else:
		return "Mixed"

def string_classes(content, classes, class1, class2, res_class):
	new_content = []
	new_classes = []
	i = 0
	while True:
		if i == len(content):
			break
		try:
			if classes[i] == class1 and classes[i+1] == class2:
				new_content.append(content[i]+' '+content[i+1])
				new_classes.append(res_class)
				classes[i+1] = 'Skip'
		except:
			print('')
		if classes[i] == 'Skip':
			i = i+1
			continue
		else:
			new_content.append(content[i])
			new_classes.append(classes[i])
		
		i = i+1

	return new_content, new_classes

def string_content(content, classes):
	print("Initial Length: "+str(len(content)))

	org_content = content

	for i in range(len(content)):
		if content[i] == '' or content[i] == ' ':
			if classes[i-1] != 'Handwriting' and classes[i+1] != 'Handwriting':
				content[i] = 'Not Mentioned'
				classes[i] = 'Handwriting'
		try:
			#CHECK 
			if classes[i] == 'Label' and classes[i+1] == 'Mixed' and classes[i-1] != 'Label':
				content[i] = content[i]+' Not Mentioned/Unrecognizable'
				classes[i] = 'Mixed'
		except:
			continue
	
	empty = [i for i in range(len(content)) if content[i] == '' or content[i] == ' ']
	content = [content[i] for i in range(len(content)) if i not in empty]
	classes = [classes[i] for i in range(len(classes)) if i not in empty]

	content, classes = string_classes(content, classes, "Handwriting", "Handwriting", "Handwriting")
	content, classes = string_classes(content, classes, "Label", "Label", "Label")
	content, classes = string_classes(content, classes, "Label", "Handwriting", "Mixed")

	content = [content[i] for i in range(len(content)) if classes[i]=='Mixed' and content[i].strip().lower() != 'none']

	all_string = ''
	for c in content:
		all_string = all_string+' '+c 

	removables = ''

	for i in range(len(org_content)):
		if org_content[i] not in all_string:
			removables = removables+' '+org_content[i]

	print("Final Length: "+str(len(content))+'\n')

	return content, classes, removables

def clean_labels(labels, removables):
	labels = [labels[i] for i in range(len(labels)) if labels[i] not in removables]
	return labels

def process_text(labels, infos, images):
	classes = [classify_block(images[i]) for i in range(len(images))]
	infos, classes, removables = string_content(infos, classes)

	labels, infos = clean_text(labels, infos)
	labels = [label for label in labels if label != '' and label != ' ']
	labels, infos = remove_words(labels, infos)

	return labels, infos