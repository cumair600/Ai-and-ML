import os,codecs,numpy
from matplotlib import pyplot as plt
import math

#Global Variables

data_dict = {}

# Functions

def make_image(image):
    plt.imshow(image,cmap="gray")
    plt.title("Sigmoid")
    plt.show()

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

def initializeW():
    w = [[0 for x in range(28)] for y in range(28)]
    return w;

def calculateDot(a,b):
    dot = []
    for i in range(0,28):
        for j in range(0,28):
            value = a[i][j] * b[i][j]
            dot.append(value)
    return sum(dot)

def multiply(value,image):
    ans = [[0 for x in range(28)] for y in range(28)]
    for i in range(0,28):
        for j in range(0,28):
            ans[i][j] = value * image[i][j]
    return ans

def add(w,result):
    ans = [[0 for x in range(28)] for y in range(28)]
    for i in range(0,28):
        for j in range(0,28):
            ans[i][j] = w[i][j] + result[i][j]
    return ans

def sigmoidFunction(x):
    return 1 / (1 + numpy.exp(-x))


def gradientDescent(w,learningRate,expected,actual,trainImage):
    ans = actual - expected
    ans = ans * (1 - expected) * expected
    result = multiply(ans,trainImage)
    result = multiply(learningRate,result)
    newW = add(w, result)
    return newW

def perceptronMultiLayeredTraining(w,trainImages,trainLabels,digit,i):
    learningRate = 0.00001
    errors = 0
    for j in range(0,len(trainImages)):
        dot = calculateDot(w,trainImages[j])
        sigmoid = sigmoidFunction(dot)
        if(sigmoid >= 0.5):
            expected = 1
        else:
            expected = 0
        if(trainLabels[j] == digit):
            actual = 1
        else:
            actual = 0
        if(expected != actual):
            errors = errors + 1
        w = gradientDescent(w,learningRate,sigmoid,actual,trainImages[j])
    print("Test Error Rate for digit-" + str(digit) + " is : " + str((errors / len(trainImages)) * 100) + " %")
    return w

# Main()

load_data()

trainImages = data_dict['train_images']
trainLabels = data_dict['train_labels']

neurons = {}

digit = 0
epoch = 1
print("Activation Function : Sigmoid")
for j in range(0,10):
    w = initializeW()
    for i in range(epoch):
        w = perceptronMultiLayeredTraining(w,trainImages,trainLabels,digit,i)
    neurons[digit] = w
    make_image(w)
    digit = digit + 1



