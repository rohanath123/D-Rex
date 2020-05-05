from ocr_test import *
import os
import matplotlib
import matplotlib.pyplot as plt

from helper import * 

PATH = "C:/Users/Rohan/Desktop/Misc Images"

def get_text_from_images(PATH):

	files = next(os.walk(PATH))[2] #dir is your directory path as string

	for file in files:
		try:
			text = detect_document(PATH+'/'+str(file))
			if text == ' ':
				continue
			else:
				print(str(file)+": "+"Length: "+str(len(text))+" Text: "+text)
		except:
			print("Error, no text.")

'''
import cv2

img = cv2.imread("D:/Deep Learning Training Data/Empty Forms/Empty Forms/0011976929.png")
plt.imshow(pil(img))
plt.show()'''