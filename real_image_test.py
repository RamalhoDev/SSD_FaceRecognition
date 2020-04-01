from PIL import Image, ImageTk
import numpy as np
import os
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tqdm import tqdm


def isOdd(n):
    if n % 2 != 0:
        return -1
    return 0 

def crop(array, newSize):
    centerX = 112//2
    centerY = 92//2

    pos = newSize//2
    return array[centerX-pos+isOdd(newSize) : centerX+pos, centerY-pos+isOdd(newSize) : centerY+pos]

# Apply FFT to an image and shifting the zero-frequency component to the center.
def getFFT(url, side):
    a = Image.open(url)
    arr = np.array(a)
    fourizado = np.fft.fft2(arr)
    fshift = np.fft.fftshift(fourizado)
    return crop(fshift, side)
            
# Initing data
original_X = []
original_y = []
side = 7

# Populating data with images
for folder in (os.listdir("./orl_faces")):
    current =[]
    for img in os.listdir(f"./orl_faces/{folder}"):
        # Getting the fft from images and changing the scale to log.
        img = getFFT(f"./orl_faces/{folder}/{img}", side)
        current.append(img)
        # Removing 's' from folder and transforming to int
    original_X.append(current)
    original_y.append(int(folder[1::]))
best_mean = 0
score = 0

size = side*side

# Transforming from list to np.array
X = np.array(original_X)
y = np.array(original_y)
X = X.reshape((40,10,size))

# Creating KNN Object
knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean', algorithm='auto', n_jobs=2)
knnSum = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean', algorithm='auto', n_jobs=2)
knnMerge = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean', algorithm='auto', n_jobs=2)
knnReal = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean', algorithm='auto', n_jobs=2)
knnImag = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean', algorithm='auto', n_jobs=2)
knnRealImag = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean', algorithm='auto', n_jobs=2)
knnImagReal = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean', algorithm='auto', n_jobs=2)

# Creating ndarray used to store training and testing data
Xs_train = np.ndarray(((X.shape)[0], 9, size), dtype=complex)
Xs_test = np.ndarray(((X.shape)[0], 1, size), dtype=complex)
ys_train = np.ndarray(((X.shape)[0], 9))
ys_test = np.ndarray(((X.shape)[0], 1))

# Applying cross validadtion for every folder
for i in (range((X.shape)[0])):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X[i], np.repeat(y[i], 10), test_size=0.1)
    Xs_train[i] = X_train
    Xs_test[i] = X_test
    ys_train[i] = y_train
    ys_test[i] = y_test
    # for idxsTrain, idxTest in loo.split(X[i]):
        # X_train, X_test, y_train, y_test = X[i][idxsTrain], X[i][idxTest], np.repeat(y[i], 9), np.array(y[i])
        # Xs_train[i] = X_train
        # Xs_test[i] = X_test
        # ys_train[i] = y_train
        # ys_test[i] = y_test
# Reshaping data to make easy when training and testing
X_train, X_test, y_train, y_test = Xs_train.reshape((40*9, size)), Xs_test.reshape((40, size)), ys_train.reshape((40*9)) ,ys_test.reshape((40))


# Training
knn.fit(X_train.real, y_train.real)
knnSum.fit(X_train.real + X_train.imag, y_train.real)
knnMerge.fit(abs(X_train), y_train.real)
knnReal.fit(X_train.real, y_train.real)
knnImag.fit(X_train.imag, y_train.real)
knnRealImag.fit(X_train.real, y_train.real)
knnImagReal.fit(X_train.imag, y_train.real)

# Predicting used to get score
y_pred = []
for x in X_test:
    predReal = knnReal.kneighbors(x.real.reshape(1, -1))
    predImag = knnImag.kneighbors(x.imag.reshape(1, -1))
    predRealImag = knnRealImag.kneighbors(x.imag.reshape(1, -1))
    predImagReal = knnImagReal.kneighbors(x.real.reshape(1, -1))
    smallest = 5000000000000
    bestIdx = 0
    for idx, i in enumerate([predReal,predImag,predRealImag,predImagReal]):
        distance = i[0].flatten()
        if distance[0] < smallest:
            smallest = distance
            bestIdx = idx

    if bestIdx == 0:
         y_pred.append(knnReal.predict(x.real.reshape(1, -1))[0])
    elif bestIdx == 1:
        y_pred.append(knnImag.predict(x.imag.reshape(1, -1))[0])
    elif bestIdx == 2:
        y_pred.append(predRealImag.predict(x.imag.reshape(1, -1))[0])
    elif bestIdx == 3:
        y_pred.append(predImagReal.predict(x.real.reshape(1, -1))[0])


y_predMerge = knnMerge.predict(abs(X_test))
y_predSum = knnSum.predict(X_test.real+X_test.imag)
y_predReal = knnReal.predict(X_test.real)
y_predImag = knnImag.predict(X_test.imag)
y_predRealImag = knnRealImag.predict(X_test.imag)
y_predImagReal = knnImagReal.predict(X_test.real)

# Labels not used in prediction
# print(set(y_test) - set(y_pred))

score = accuracy_score(y_test, y_pred)
scoreSum = accuracy_score(y_test, y_predMerge)
scoreMerge = accuracy_score(y_test, y_predMerge)
scoreReal = accuracy_score(y_test, y_predReal)
scoreImag = accuracy_score(y_test, y_predImag) 
scoreRealImag = accuracy_score(y_test, y_predRealImag)
scoreImagReal = accuracy_score(y_test, y_predImagReal)


print("Real:",scoreReal, sep="\t")
print("Imag:",scoreImag, sep="\t")
print("RealImag:",scoreRealImag, sep="\t")
print("ImagReal:",scoreImagReal, sep="\t")
print("Sum:",scoreSum, sep="\t")
print("Merge:",scoreMerge, sep="\t")
print("Min:",score, sep="\t")

# Scores
# print(accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred, zero_division=0))
# print(confusion_matrix(y_test, y_pred))
