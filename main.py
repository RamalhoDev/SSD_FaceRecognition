from PIL import Image, ImageTk
import numpy as np
import os
from sklearn import model_selection

# Receive a image and return n*log(image)
def applyLogScale(fft, n = 10):
    magnitude_spectrum = n*np.log(np.abs(fft))
    return magnitude_spectrum

# Apply FFT to an image and shifting the zero-frequency component to the center.
def getFFT(url):
    a = Image.open(url)
    arr = np.array(a)
    fourizado = np.fft.fft2(arr)
    fshift = np.fft.fftshift(fourizado)
    return fshift

def getSubRegion(image, newHeight, newWidth):
    width, height = image.size
    left = (width - newWidth)//2
    right = (width + newWidth)//2
    top = (height - newHeight)//2
    bottom = (height + newHeight)//2

    return image.crop((left,top,right,bottom))

def getCropedImages(images, size):
    allImages = []

    for image in images:
        a = Image.fromarray(image.astype(np.int32))
        allImages.append(getSubRegion(a,size,size))

    return allImages
            

# Initing data
X = []
Y = []

# Populating data with images
for folder in os.listdir("./orl_faces"):
    current = []
    for img in os.listdir(f"./orl_faces/{folder}"):
        # Getting the fft from images and changing the scale to log.
        current.append(applyLogScale(getFFT(f"./orl_faces/{folder}/{img}")))
    X.append(current)
    # Removing 's' from folder and transforming to int
    Y.append(int(folder[1::]))

# Transforming from list to np.array
X = np.array(X)
Y = np.array(Y)

# Exemplo de como rodar o subregion
# croped = []
# for images in X:
#     croped = getCropedImages(images, 2)

# print(len(croped))

# Applying LeaveOneOut is like KFold where k = n.
loo = model_selection.LeaveOneOut()
# for idxTrain, idxTest in loo.split(X):
    # print("%s %s" % (X[idxTrain], X[idxTest]))
