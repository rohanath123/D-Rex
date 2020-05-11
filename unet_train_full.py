'''
PLEASE IGNORE THIS SCRIPT IT IS MEANT FOR TESTING IN DEVELOPMENT FUNCTIONS AND WILL NOT BE PUT INTO PRODUCTION 

'''


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

from google.cloud import vision

tens = transforms.ToTensor()
pil = transforms.ToPILImage()
resize = transforms.Resize((512, 512))
gray = transforms.Grayscale(3)

transforms_set = transforms.Compose([transforms.Grayscale(3), transforms.Resize((512, 512)), transforms.ToTensor()])

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

'''images = torchvision.datasets.ImageFolder("/content/drive/My Drive/Forms Dataset Created Remastered/Forms Folder", transform = transforms_set)
images = [tens(resize(pil(images[i][0]))) for i in range(len(images))]
with open("/content/drive/My Drive/Pickled Datasets/Training Dataset Full/images_pkl.txt", "wb")as p:
  pickle.dump(images, p)'''

'''masks = torchvision.datasets.ImageFolder("/content/drive/My Drive/Forms Dataset Created Remastered/Masks Folder", transform = transforms.ToTensor())
masks = [tens(resize(pil(masks[i][0]))) for i in range(len(masks))]
masks = [threshold(mask, 0.5) for mask in masks]
with open("/content/drive/My Drive/Pickled Datasets/Training Dataset Full/masks_pkl.txt", "wb")as p:
  pickle.dump(masks, p)

with open("/content/drive/My Drive/Pickled Datasets/Training Dataset Full/images_pkl.txt", "rb")as p:
  images = pickle.load(p)
with open("/content/drive/My Drive/Pickled Datasets/Training Dataset Full/masks_pkl.txt", "rb")as p:
  masks = pickle.load(p)

import sklearn
from sklearn.model_selection import train_test_split

images_train, images_test, masks_train, masks_test = train_test_split(images, masks, test_size = 0.1)
train_set = list(zip(images_train, masks_train))
trainloader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle = True)
'''
class Down(nn.Module):
  def __init__(self, input_size, output_size):
    super(Down, self).__init__()

    self.input_size = input_size
    self.output_size = output_size
    
    self.conv1 = nn.Conv2d(in_channels = self.input_size, out_channels = self.output_size, kernel_size = 3, stride = 1, padding = 1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels = self.output_size, out_channels = self.output_size, kernel_size = 3, stride = 1, padding = 1)
    self.relu2 = nn.ReLU()

    self.maxp = nn.MaxPool2d(2)

    self.drop = nn.Dropout(0.5)

  def forward(self, input):
    mid = self.relu2(self.conv2(self.relu1(self.conv1(input))))
    output = self.maxp(mid)
    output = self.drop(output)
    return output, mid


class Up(nn.Module):
  def __init__(self, input_size, output_size):
    super(Up, self).__init__()

    self.input_size = input_size
    self.output_size = output_size

    self.up = nn.ConvTranspose2d(in_channels = self.input_size, out_channels = self.output_size, kernel_size = 2, stride = 2)

    self.drop = nn.Dropout(0.5)

    self.conv1 = nn.Conv2d(in_channels= self.input_size, out_channels= self.output_size, kernel_size = 3, stride = 1, padding = 1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels= self.output_size, out_channels= self.output_size, kernel_size = 3, stride = 1, padding = 1)
    self.relu2 = nn.ReLU()
    

  def forward(self, input, mid):
    output = self.up(input)
    output = self.drop(output)
    output = torch.cat([output, mid], dim = 1)
    output = self.relu2(self.conv2(self.relu1(self.conv1(output))))
    return output


class UNetModel(nn.Module):
  def __init__(self):
    super(UNetModel, self).__init__()

    self.down1 = Down(3, 32)
    self.down2 = Down(32, 64)
    self.down3 = Down(64, 128)
    self.down4 = Down(128, 256)
    #self.down5 = Down(256, 512)

    self.conv_in1 = nn.Conv2d(256, 512, 3, 1, 1)
    self.relu_in1 = nn.ReLU()
    self.conv_in2 = nn.Conv2d(512, 512, 3, 1, 1)
    self.relu_in2 = nn.ReLU()

    self.up1 = Up(512, 256)
    self.up2 = Up(256, 128)
    self.up3 = Up(128, 64)
    self.up4 = Up(64, 32)

    self.conv_last = nn.Conv2d(32, 1, 3, 1, 1)
    self.sig = nn.Sigmoid()
  
  def forward(self, input):

    out1, mid1 = self.down1(input)
    out2, mid2 = self.down2(out1)
    out3, mid3 = self.down3(out2)
    out4, mid4 = self.down4(out3)

    output = self.relu_in2(self.conv_in2(self.relu_in1(self.conv_in1(out4))))

    output = self.up1(output, mid4)
    output = self.up2(output, mid3)
    output = self.up3(output, mid2)
    output = self.up4(output, mid1)

    #output = self.conv_last(output)
    
    output = self.sig(self.conv_last(output))
    return output

def train(trainloader, learning_rate, criterion, optimizer, epochs):
  for epoch in range(epochs):
    avg_loss = 0
    #print('Epoch: ', epoch+1)
    for i, (image, label) in enumerate(trainloader):
      image = Variable(image).cuda()
      label = Variable(label).cuda()

      optimizer.zero_grad()
      
      output = model(image.view(1, 3, 512, 512))
      loss = criterion(output.view(1, 1, 512, 512), label[0][0].view(1, 1, 512, 512))
      avg_loss = avg_loss + loss.item()

      loss.backward()

      optimizer.step()
      
      #print("Epoch: ", epoch+1, "Iteration: ", i+1, "Loss: ", loss.item())
    print("Epoch: ", epoch+1, "Average Loss = ", float(avg_loss/len(images)))

pretrained = True
model = UNetModel()
if pretrained:
  model.load_state_dict(torch.load('D:/Deep Learning Trained Models/Forms/100.pt'))
model = model.cuda()

'''learning_rate = 0.0001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train(trainloader, learning_rate, criterion, optimizer, 100)
'''

import random
def test(images_test, masks_test, model):
  i = random.randint(0, len(images_test)-1)
  input = images_test[i]

  output = model(input.cuda().view(1, 3, 512, 512))
  output = threshold(output, 0.5)

  plt.imshow(pil(input))
  plt.show()

  plt.imshow(pil(output.cpu().view(1, 512, 512)), cmap = 'gray')
  plt.show()

  plt.imshow(pil(masks_test[i]))
  plt.show()

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

def custom_predict(original_image, threshold_image, threshold_output):
  custom_resize = transforms.Resize((int(round_up(tens(original_image).size()[1], -2)), int(round_up(tens(original_image).size()[2], -2))))
  input = tens(gray(resize(original_image)))
  if threshold_image:
    input = threshold(input, 0.4)
  output = model(input.cuda().view(1, 3, 512, 512))
  output = output.cpu().view(1, 512, 512)
  if threshold_output:
    output = threshold(output, 0.5)
  image = tens(custom_resize(original_image))
  output = tens(custom_resize(pil(output)))
  x, y = get_boxes(image[0], output[0])
  return x
  boxes = []
  for box in x:
    boxes.append(pil(torch.FloatTensor([box])))
  return boxes

image = Image.open("D:/Deep Learning Training Data/Forms Dataset/images/images/5.jpg")
boxes = custom_predict(image, False, False)

from ocr_test import detect_document

for box in boxes:
  detect_document(box.tobytes())
