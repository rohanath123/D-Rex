#FILE FOR DISTRIBUTED WRITING: CREATING IMAGES OF RANDOM SENTENCES TO MIMIC HANDWRITING DISTRIBUTION IN FORMS

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

import matplotlib
import matplotlib.pyplot as plt

import random

pil = transforms.ToPILImage()
tens = transforms.ToTensor()

def is_collide(img, blank_img, x, y):
  if torch.all(torch.eq(blank_img[0][x:x+img.size()[1], y:y+img.size()[2]], torch.ones(1, img.size()[1], img.size()[2]))):
    return False
  else:
    return True 

def stitch_images(imgs, max_size):
  blank_img = torch.ones((3, max_size, max_size))
  for img in imgs:
    while True:
      x = random.randint(1, max_size-img.size()[1])
      y = random.randint(1, max_size-img.size()[2])

      if is_collide(img, blank_img, x, y) == False:
        for i in range(3):
          blank_img[i][x:x+img.size()[1], y:y+img.size()[2]] = img[i]
        break

  return blank_img

def make_forms(imgs, size, safety_size):
  forms = []
  max_size = round(max([imgs[i].size()[2] for i in range(len(imgs))]) + safety_size, -3)

  for i in range(size):
    no_sentences = random.randint(4, 6)
    sentences = random.sample(imgs, no_sentences)
    forms.append(stitch_images(sentences, max_size))
    
  return forms

def save_forms_set(forms, PATH):
  for i in range(len(forms)):
    pil(forms[i]).save(PATH+str(i)+'.png')

def make_and_save_forms(imgs, size, PATH):
  forms = make_forms(imgs, size)
  save_forms_set(forms, PATH, num)
  del forms


