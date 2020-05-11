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
      
    print("Epoch: ", epoch+1, "Average Loss = ", float(avg_loss/len(images)))

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


learning_rate = 0.0001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#UNCOMMENT FOLLOWING LINE TO MANUALLY TRAIN FRESH MODEL
#train(trainloader, learning_rate, criterion, optimizer, 100)

