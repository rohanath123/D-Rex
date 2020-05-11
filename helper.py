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

to_remove = ['-', '"', ':', '.']

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
  return white_sections_img, white_sections_msk

def get_boxes(img, msk):
  img_boxes = []
  msk_boxes = []
  img_sections, msk_sections = get_white_sections(img, msk)
  print(len(img_sections))

  for i in range(len(msk_sections)):
    img_box, msk_box = get_white_sections(np.transpose(img_sections[i]), np.transpose(msk_sections[i]))
    #print(len(img_box))
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
    boxes.append(pil(torch.FloatTensor([box])))
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

def get_prediction_image(org_image):
  image = remove_lines(org_image)
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

def isolate_printed_text(image):
  image = tens(image)
  image = image != 0
  image = image.float()
  return image

def get_cleaned_text(PATH, image, isolate):
  if isolate:
    image = isolate_printed_text(image)
    image = pil(image)

  image.save(PATH)

  text = detect_document(PATH)
  text = ''.join([c for c in text if c not in to_remove])

  if ''.join(set(text)) == ' ':
    return ' '
  else:
    return text