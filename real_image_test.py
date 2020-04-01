from PIL import Image, ImageTk
import numpy as np
import os
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tqdm import tqdm

def getSubRegion(image, newHeight, newWidth):
    width, height = image.size
    left = (width - newWidth)//2
    right = (width + newWidth)//2
    top = (height - newHeight)//2
    bottom = (height + newHeight)//2

    return image.crop((left,top,right,bottom))

# Apply FFT to an image and shifting the zero-frequency component to the center.
def getFFT(url, side):
    a = Image.open(url)
    arr = np.array(getSubRegion(a, side, side))
    fourizado = np.fft.fft2(arr)
    fshift = np.fft.fftshift(fourizado)
    return fshift

def getCroppedImages(images, size):
    allImagesArray = []

    for image in images:
        a = Image.fromarray(image)
        allImagesArray.append(np.array(getSubRegion(a,size,size)))

    return allImagesArray
            

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
local_score = []
size = side*side

for time in tqdm(range(0, 10), desc=f"RUN"):
    # Transforming from list to np.array
    X = np.array(original_X)
    y = np.array(original_y)
    X = X.reshape((40,10,size))

    # Creating KNN Object
    knn = KNeighborsClassifier(n_neighbors=1, metric = 'euclidean', weights='uniform', algorithm='auto', n_jobs=2)

    # Creating ndarray used to store training and testing data
    Xs_train = np.ndarray(((X.shape)[0], 9, size), dtype=complex)
    Xs_test = np.ndarray(((X.shape)[0], 1, size), dtype=complex)
    ys_train = np.ndarray(((X.shape)[0], 9), dtype=complex)
    ys_test = np.ndarray(((X.shape)[0], 1), dtype=complex)

    # Applying kfold for every folder
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
    print(X_train.real, "\n", X_train.imag)
    input()
    knn.fit(X_train, y_train)

    # Predicting used to get scores
    y_pred = knn.predict(X_test)

    # Labels not used in prediction
    # print(set(y_test) - set(y_pred))
    score = accuracy_score(y_test, y_pred)
    # Scores
    # print(accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred, zero_division=0))
    # print(confusion_matrix(y_test, y_pred))
    local_score.append(score)
scores = np.array(local_score)
print(scores)
print(scores.mean())

# Exemplo de como rodar o subregion
# croped = []
# for images in X:
#     croped = getCropedImages(images, 2)

# print(len(croped))