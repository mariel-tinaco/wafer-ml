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
source_path = './ADC_Dataset/ADC_Dataset/preproccessed/'
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
        den = cv2.fastNlMeansDenoising(np.asarray(img),None,7,11,81)

        
        # Contrast Enhancer
        enhancer = ImageEnhance.Contrast(Image.fromarray(den))
        img_gray = enhancer.enhance(2)
        
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


        # Get Wavelet Components
        den = Image.fromarray(den)
        coeffs = pywt.dwt2(data=img_gray, wavelet='db6', mode='sym')
        cA, (cH, cV, cD) = coeffs

        coeffs2 = pywt.dwt2(data=cA, wavelet='db6', mode='sym')
        cA2, (cH2, cV2, cD2) = coeffs2
        
        # Concatenate to coefficients to a single Array
        # 1st Level DWT
        coeff1 = np.vstack((np.hstack((cA,cH)),np.hstack((cV,cD))))

        
        # 2nd Level DWT
        coeff2 = np.vstack((np.hstack((cA2,cH2)),np.hstack((cV2,cD2))))
        
        
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(coeff1,cmap="gray")
        plt.title('Level-1 DWT')
        plt.subplot(1,2,2)
        plt.imshow(coeff1,cmap="gray")
        plt.title('Level-2 DWT')

        '''

        plt.figure(1)
        #Approximation
        plt.subplot(2,2,1)
        plt.imshow(cA,cmap="gray")
        plt.title('Approx')

        #Horizontal 
        plt.subplot(2,2,2)
        plt.imshow(cH,cmap="gray")
        plt.title('H-Array')

        #Vertical        
        plt.subplot(2,2,3)
        plt.imshow(cV,cmap="gray")
        plt.title('V-Array')

        #Diagonal
        plt.subplot(2,2,4)
        plt.imshow(cD,cmap="gray")
        plt.title('D-Array')

        plt.figure(2)
        #Approximation
        plt.subplot(2,2,1)
        plt.imshow(cA2,cmap="gray")
        plt.title('Approx')

        #Horizontal 
        plt.subplot(2,2,2)
        plt.imshow(cH2,cmap="gray")
        plt.title('H-Array')

        #Vertical        
        plt.subplot(2,2,3)
        plt.imshow(cV2,cmap="gray")
        plt.title('V-Array')

        #Diagonal
        plt.subplot(2,2,4)
        plt.imshow(cD2,cmap="gray")
        plt.title('D-Array')'''

        
        cH = np.zeros_like(cH,dtype='uint8')
        cV = np.zeros_like(cV,dtype='uint8')
        cH2 = np.zeros_like(cH2,dtype='uint8')
        cV2 = np.zeros_like(cV2,dtype='uint8')

        cA = pywt.idwt2(coeffs=(cA2, (cH2, cV2, cD2)),wavelet='db6', mode='sym')
        cA = cA[0:len(cH),0:len(cH)]
        img_rec = pywt.idwt2(coeffs=(cA, (cH, cV, cD)),wavelet='db6', mode='sym')

        
        plt.figure(0)
        plt.subplot(2,2,4)
        plt.imshow(img_rec,cmap = 'gray')
        plt.title('Reconstructed with 0 H&V')


        '''
        # PROCESS HORIZONTAL COMPONENT
        plt.figure(3)
        plt.subplot(3,3,1)
        plt.imshow(cH,cmap='gray')
        plt.title('H-Array')
        cH_bin = np.asarray(cH, dtype='uint8')
        cH_bin = Image.fromarray(cH_bin)
        cH_bin = cH_bin.filter(ImageFilter.FIND_EDGES)
        cH_bin = cH_bin.filter(ImageFilter.EDGE_ENHANCE)
        #cH_bin = cH_bin.filter(ImageFilter.EDGE_ENHANCE_MORE)
        cH_bin = cH_bin.filter(ImageFilter.SMOOTH)
        cH_bin = cH_bin.filter(ImageFilter.SMOOTH_MORE)

        # Remove Vertical lines on vertical component of DWT
        cH_bin = np.asarray(cH_bin, dtype='uint8')
        cH_bin = cv2.adaptiveThreshold(cH_bin, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

        plt.subplot(3,3,2)
        plt.imshow(cH_bin,cmap='gray')
        plt.title('Binary Edge of H-Array')
        
        horizontal = cH_bin
        horizontal_size = int(np.shape(horizontal)[1] / 30) # size of 8
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontal_size,1))
        horizontal = cv2.erode(horizontal,horizontalStructure,iterations=2)
        horizontal = cv2.dilate(horizontal,horizontalStructure,iterations=7)
        
        plt.subplot(3,3,3)
        plt.imshow(horizontal,cmap='gray')
        plt.title('Lines on H-Array')


        # PROCESS VERTICAL COMPONENT
        plt.subplot(3,3,4)
        plt.imshow(cV,cmap='gray')
        plt.title('V-Array')
        cV_bin = np.asarray(cV, dtype='uint8')
        cV_bin = Image.fromarray(cV_bin)
        cV_bin = cV_bin.filter(ImageFilter.FIND_EDGES)
        cV_bin = cV_bin.filter(ImageFilter.EDGE_ENHANCE)
        #cV_bin = cV_bin.filter(ImageFilter.EDGE_ENHANCE_MORE)
        cV_bin = cV_bin.filter(ImageFilter.SMOOTH)
        cV_bin = cV_bin.filter(ImageFilter.SMOOTH_MORE)


        # Remove Vertical lines on vertical component of DWT
        cV_bin = np.asarray(cV_bin, dtype='uint8')
        cV_bin = cv2.adaptiveThreshold(cV_bin, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY , 7, 2)

        plt.subplot(3,3,5)
        plt.imshow(cV_bin,cmap='gray')
        plt.title('Binary Edge of V-Array')
        
        vertical = cV_bin
        vertical_size = int(np.shape(vertical)[1] / 30) # size of 8
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,vertical_size))
        vertical = cv2.erode(vertical,verticalStructure,iterations=1)
        vertical = cv2.dilate(vertical,verticalStructure,iterations=5)
        
        plt.subplot(3,3,6)
        plt.imshow(vertical,cmap='gray')
        plt.title('Lines on V-Array')


        # PROCESS DIAGONAL COMPONENT
        plt.subplot(3,3,7)
        plt.imshow(cD,cmap='gray')
        plt.title('D-Array')
        cD_bin = np.asarray(cD, dtype='uint8')
        cD_bin = Image.fromarray(cD_bin)
        cD_bin = cD_bin.filter(ImageFilter.FIND_EDGES)
        cD_bin = cD_bin.filter(ImageFilter.EDGE_ENHANCE)
        #cD_bin = cD_bin.filter(ImageFilter.EDGE_ENHANCE_MORE)
        cD_bin = cD_bin.filter(ImageFilter.SMOOTH)
        cD_bin = cD_bin.filter(ImageFilter.SMOOTH_MORE)

        # Remove Vertical lines on vertical component of DWT
        cD_bin = np.asarray(cD_bin, dtype='uint8')
        cD_bin = cv2.adaptiveThreshold(cD_bin, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)

        plt.subplot(3,3,8)
        plt.imshow(cD_bin,cmap='gray')
        plt.title('Binary Edge of D-Array')
        
        diagonal = cD_bin
        diagonal_size = int(np.shape(vertical)[1] / 30) # size of 8
        diagonalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(diagonal_size,diagonal_size))
        diagonal = cv2.erode(diagonal,diagonalStructure,iterations=1)
        diagonal = cv2.dilate(diagonal,diagonalStructure,iterations=1)
        
        plt.subplot(3,3,9)
        plt.imshow(diagonal,cmap='gray')
        plt.title('Lines on D-Array')'''
        
        plt.show()