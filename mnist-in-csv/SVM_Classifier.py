import os,codecs,numpy
from matplotlib import pyplot as plt
import math
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

#Global Variables

data_dict = {}

# Functions

def get_int(b):   # CONVERTS 4 BYTES TO A INT
    return int(codecs.encode(b, 'hex'), 16)

def load_data():
    # PROVIDE YOUR DIRECTORY WITH THE EXTRACTED FILES HERE
    datapath = 'MNIST DataSet/'
    files = os.listdir(datapath)

    for file in files:
        if file.endswith('ubyte'):  # FOR ALL 'ubyte' FILES
            print('Reading ',file)
            with open (datapath+file,'rb') as f:
                data = f.read()
                type = get_int(data[:4])   # 0-3: THE MAGIC NUMBER TO WHETHER IMAGE OR LABEL
                length = get_int(data[4:8])  # 4-7: LENGTH OF THE ARRAY  (DIMENSION 0)
                if (type == 2051):
                    category = 'images'
                    num_rows = get_int(data[8:12])  # NUMBER OF ROWS  (DIMENSION 1)
                    num_cols = get_int(data[12:16])  # NUMBER OF COLUMNS  (DIMENSION 2)
                    parsed = numpy.frombuffer(data,dtype = numpy.uint8, offset = 16)  # READ THE PIXEL VALUES AS INTEGERS
                    parsed = parsed.reshape(length,num_rows,num_cols)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES x HEIGHT x WIDTH]
                elif(type == 2049):
                    category = 'labels'
                    parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=8) # READ THE LABEL VALUES AS INTEGERS
                    parsed = parsed.reshape(length)  # RESHAPE THE ARRAY AS [NO_OF_SAMPLES]
                if (length==10000):
                    set = 'test'
                elif (length==60000):
                    set = 'train'
                data_dict[set+'_'+category] = parsed  # SAVE THE NUMPY ARRAY TO A CORRESPONDING KEY

def calc_hog_features(X, image_shape=(28, 28), pixels_per_cell=(8, 8)):
    fd_list = []
    for row in X:
        img = row.reshape(image_shape)
        fd = hog(img, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1))
        fd_list.append(fd)

    return numpy.array(fd_list)

# Main()

load_data()

trainImages = data_dict['train_images']
testImages = data_dict['test_images']
trainLabels = data_dict['train_labels']
testLabels = data_dict['test_labels']

X_train = calc_hog_features(trainImages, pixels_per_cell=(8, 8))
X_test = calc_hog_features(testImages, pixels_per_cell=(8, 8))

clf = svm.LinearSVC()
clf.fit(X_train, trainLabels)

result = clf.predict(X_test)

errorCount = 0
for i in range(0, len(result)):
	if result[i] != testLabels[i]:
		errorCount += 1
print('Test Error Rate For SVM Classifier : ' + str((errorCount/len(testImages)*100)))



