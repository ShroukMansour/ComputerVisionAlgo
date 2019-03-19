import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.svm import SVC
from HOG.hog import apply_hog
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])

# TODO: Data augmentation methods


def get_training_data(data_path):
    X = []; y = []
    img_files = listdir(data_path+"human")
    for i in range(len(img_files)):
        X.append(apply_hog(data_path +"human/" + img_files[i]))
        y.append(1)
    img_files = listdir(data_path + "test")
    for i in range(len(img_files)):
        X.append(apply_hog(data_path +"test/" + img_files[i]))
        y.append(0)
    return np.array(X), np.array(y)


def shuffle_data(X, y):
    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    return X[s], y[s]

def predict_img(path):
    x = apply_hog(path)
    print(clf.predict([x]))


X, y = get_training_data("/home/shrouk_mansour/Downloads/HOG dataset/")
X, y = shuffle_data(X, y)
clf = SVC(gamma='auto', kernel='linear')
clf.fit(X, y)


# print(clf.predict(X))
print(clf.score(X, y))
predict_img("/home/shrouk_mansour/Pictures/1.jpg")

