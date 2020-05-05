from ocr_test import *
import os
import matplotlib
import matplotlib.pyplot as plt

from helper import * 

def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
                    content = image_file.read()
    image = vision.types.Image(content=content)

    response = client.document_text_detection(image=image)
    total = ' '
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    total = total+word_text+" "

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return total.lower()

PATH = "C:/Users/Rohan/Desktop/Misc Images"

def get_text_from_images(PATH):
    files = next(os.walk(PATH))[2]
    for file in files:
        try:
            text = detect_document(PATH+'/'+str(file))
            if text == ' ':
                continue
            else:
                print(str(file)+": "+"Length: "+str(len(text))+" Text: "+text)
        except:
            print("Error, no text.")
