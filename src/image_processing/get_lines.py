import pywt
import os      
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import cv2
import numpy as np

def binarize(img):
    try:
        img = Image.fromarray(img)
    except Exception as e:
        print(e)
    #initialize threshold
    thresh=5

    #convert image to greyscale
    img=img.convert('L') 

    width,height=img.size

    #traverse through pixels 
    for x in range(width):
        for y in range(height):
            #if intensity less than threshold, assign white
            if img.getpixel((x,y)) < thresh:
                img.putpixel((x,y),0)

            #if intensity greater than threshold, assign black 
            else:
                img.putpixel((x,y),255)

    return img

cwd = os.getcwd()
source_path = './wafer-ml/ADC_Dataset/train/'
categories = [
        'CMPMicroscratch',
        'CrystalDislocation',
        'DAP',
        'EnclosedDefect',
        'EtchBlock',
        'Fiber',
        'Flake',
        'MissingTrenchFill',
        'NonVisual',
        'Particle',
        'PolyNodules',
        'Residue'
    ]

for cats in categories: 
    fnames = os.listdir(source_path+cats)

    for i in range(15,16):
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["figure.autolayout"] = True
        img = Image.open(source_path+cats+'/'+fnames[i])
        img = ImageOps.grayscale(img)

        # Denoising
        den = cv2.fastNlMeansDenoising(np.asarray(img),None,10,11,51)

        
        # Contrast Enhancer
        enhancer = ImageEnhance.Contrast(Image.fromarray(den))
        img_gray = enhancer.enhance(2)
        

        # Using Denoised Image
        img_bin = np.asarray(den, dtype='uint8')
        img_bin = Image.fromarray(img_bin)
        img_bin = img_bin.filter(ImageFilter.FIND_EDGES)
        #img_bin = img_bin.filter(ImageFilter.EDGE_ENHANCE)
        #img_bin = img_bin.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img_bin = img_bin.filter(ImageFilter.SMOOTH)
        #img_bin = img_bin.filter(ImageFilter.SMOOTH_MORE)


        img_bin = np.asarray(img_bin, dtype='uint8')
        img_bin = cv2.adaptiveThreshold(img_bin, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        
        plt.figure(0)
        plt.subplot(2,2,1)
        plt.imshow(img,cmap='gray')
        plt.title(cats + ' '+fnames[i])
        plt.subplot(2,2,2)
        plt.imshow(den,cmap='gray')
        plt.title('Denoised')
        plt.subplot(2,2,3)
        plt.imshow(img_gray,cmap='gray')
        plt.title('Enhanced')
        plt.subplot(2,2,4)
        plt.imshow(img_bin,cmap='gray')
        plt.title('Binary')

        
        # Get the LINES
        # PROCESS HORIZONTAL COMPONENT
        horizontal = img_bin
        horizontal_size = int(np.shape(horizontal)[1] / 50) # size of 8
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontal_size,1))
        horizontal = cv2.erode(horizontal,horizontalStructure,iterations=2)
        horizontal = cv2.dilate(horizontal,horizontalStructure,iterations=7)
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(horizontal,cmap='gray')
        plt.title('Horizontal Lines')
        

        # PROCESS VERTICAL COMPONENT
        vertical = img_bin
        vertical_size = int(np.shape(vertical)[1] / 50) # size of 8
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,vertical_size))
        vertical = cv2.erode(vertical,verticalStructure,iterations=2)
        vertical = cv2.dilate(vertical,verticalStructure,iterations=7)
        
        plt.subplot(1,2,2)
        plt.imshow(vertical,cmap='gray')
        plt.title('Vertical Lines')
        
        plt.show()