from PIL import Image, ImageTk
import numpy as np
import os



def getFFT(url):
    a = Image.open(url)
    # a = Image.open("./orl_faces/s1/1.pgm")
    arr = np.array(a)
    fourizado = np.fft.fft2(arr)
    fshift = np.fft.fftshift(fourizado.real)
    magnitude_spectrum = 15*np.log(np.abs(fshift))
    magnitude_spectrum = np.round(magnitude_spectrum)
    return Image.fromarray(magnitude_spectrum.astype(int))

a = getFFT("./orl_faces/s1/1.pgm")
a.show()

# images = []
# for folder in os.listdir("./orl_faces"):
#     current = []
#     for img in os.listdir(f"./orl_faces/{folder}"):
#         current.append(getFFT(f"./orl_faces/{folder}/{img}"))
#     images.append(current)