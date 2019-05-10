import os,codecs,numpy
import math

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


def subtractNSquare(a,b):
    ss = []
    for i in range(0,28):
        for j in range(0,28):
            ans = a[i][j] - b[i][j]
            ans = ans**2
            ss.append(ans)
    return ss


def findEuclideanDistance(a, b):
    euclidean_distance = subtractNSquare(a,b)
    euclidean_distance = sum(euclidean_distance)
    euclidean_distance = math.sqrt(euclidean_distance)
    return euclidean_distance

def getAllEDs(test,train):
    EDs = []
    for i in range(0,60000):
        euclideanDistance = findEuclideanDistance(test,train[i])
        EDs.append(euclideanDistance)
    return EDs

def getK_EDs(EDs,k):
    minIndices = []
    for i in range(0,k):
        index = numpy.argmin(EDs)
        minIndices.append(index)
        EDs[index] = EDs[index]*EDs[index] # replacing
    return minIndices

def findMagicNo(train,minIndices,k):
    magic = [0,0,0,0,0,0,0,0,0,0]
    for i in range(0,k):
        value = train[minIndices[i]]
        magic[value] = magic[value] + 1
    return numpy.argmax(magic)

#Main()

load_data()
k = 100
noOfErrors = 0
testDataSize = 300

for i in range(0,testDataSize):
    a = data_dict['test_images'][i]
    b = data_dict['train_images']
    EDs = getAllEDs(a,b)
    minIndices = getK_EDs(EDs,k)
    c = data_dict['test_labels'][i]
    d = data_dict['train_labels']
    magicNo = findMagicNo(d,minIndices,k)
    print('Actual Label is : '+ str(c))
    print('Predicted Label is : ' + str(magicNo))
    if c != magicNo:
        noOfErrors = noOfErrors + 1

print("Error Percentage is : " + str((noOfErrors/testDataSize)*100))
