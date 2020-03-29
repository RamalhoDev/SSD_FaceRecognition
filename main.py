from PIL import Image, ImageTk
import numpy as np
import os
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tqdm import tqdm
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

# Initing data
X = []
y = []

# Populating data with images
for folder in (os.listdir("./orl_faces")):
    current =[]
    for img in os.listdir(f"./orl_faces/{folder}"):
        # Getting the fft from images and changing the scale to log.
        current.append(applyLogScale(getFFT(f"./orl_faces/{folder}/{img}")))
        # Removing 's' from folder and transforming to int
    X.append(current)
    y.append(int(folder[1::]))

# Transforming from list to np.array
X = np.array(X)
y = np.array(y)
size = 112*92
X = X.reshape((40,10,size))
# Applying LeaveOneOut is like KFold where k = n.
loo = model_selection.LeaveOneOut()
# Creating KNN Object
knn = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', n_jobs=2)

# Creating ndarray used to store training and testing data
Xs_train = np.ndarray(((X.shape)[0], 9, size))
Xs_test = np.ndarray(((X.shape)[0], 1, size))
ys_train = np.ndarray(((X.shape)[0], 9))
ys_test = np.ndarray(((X.shape)[0], 1))


# Applying kfold for every folder
for i in (range((X.shape)[0])):
    for idxsTrain, idxTest in loo.split(X[i]):
        X_train, X_test, y_train, y_test = X[i][idxsTrain], X[i][idxTest], np.repeat(y[i], 9), np.array(y[i])
        Xs_train[i] = X_train
        Xs_test[i] = X_test
        ys_train[i] = y_train
        ys_test[i] = y_test

# Reshaping data to make easy training and testing
X_train, X_test, y_train, y_test = Xs_train.reshape((40*9, size)), Xs_test.reshape((40, size)), ys_train.reshape((40*9)) ,ys_test.reshape((40))

# Training
knn.fit(X_train, y_train)

# Predicting used to get scores
y_pred = knn.predict(X_test)

# Labels not used in prediction
print(set(y_test) - set(y_pred))

# Scores
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
print(confusion_matrix(y_test, y_pred))