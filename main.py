from PIL import Image, ImageTk
import numpy as np
import os

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

sizes = [2,4,6,10,12,15,20,25,30,30,50]
a = getFFT("orl_faces/s1/1.pgm")
a = Image.fromarray(a.astype(np.int32))

cropImages = []

for i in sizes:
    b = getSubRegion(a, i, i)
    cropImages.append(b)

# width, height = b.size
# print(f"{width} {height}")

# images = []
# for folder in os.listdir("./orl_faces"):
#     current = []
#     for img in os.listdir(f"./orl_faces/{folder}"):
#         current.append(getFFT(f"./orl_faces/{folder}/{img}"))
#     images.append(current)