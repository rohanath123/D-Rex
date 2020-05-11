import torch
import torchvision 
from torchvision import transforms

import numpy as np 
from PIL import Image

import pickle

from helper import *

def get_raw_data():
  images = torchvision.datasets.ImageFolder("< PATH TO REMASTERED FORMS IMAGES SET  >", transform = transforms_set)
  images = [tens(resize(pil(images[i][0]))) for i in range(len(images))]

  masks = torchvision.datasets.ImageFolder("< PATH TO REMASTERED FORMS MASKS SET  >", transform = transforms.ToTensor())
  masks = [tens(resize(pil(masks[i][0]))) for i in range(len(masks))]
  masks = [threshold(mask, 0.5) for mask in masks]

  return images, masks

def split_data(images, masks):
  images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size = 0.1)
  train_set = list(zip(images_train, masks_train))
  trainloader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle = True)

  return trainloader

def pickle_data(images, masks):
  with open("< PATH TO DUMP PICKLED DATASET IMAGES >"+".txt", "wb")as p:
    pickle.dump(images, p)

  with open("< PATH TO DUMP PICKLED DATASET MASKS >"+".txt", "wb")as p:
    pickle.dump(masks, p)

def load_pickled_data():
  with open("< PATH TO PICKLED DATASET IMAGES >, "rb")as p:
    images = pickle.load(p)

  with open("< PATH TO DUMP PICKLED DATASET MASKS >, "rb")as p:
    masks = pickle.load(p)

  return images, masks



