import numpy
import matplotlib.pyplot as plt
import idx2numpy

#Converts the MNIST Binary dataset into a grey scale format
train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
#ensures that the output stays on one line in the terminal (only if it should)
numpy.set_printoptions(linewidth=numpy.inf)
numpy.set_printoptions(suppress=True)

def reshape(train_images):
    new_train_images = numpy.array(train_images[0]).reshape(1,784)
    for i in range(1,1000):
        new_train_images = numpy.append(new_train_images,train_images[i].reshape(1,784),axis=0)
    return new_train_images

    #new_train_images = train_images.reshape(1000,784)
    #return numpy.array(new_train_images,dtype=float)

#encodes all the greyscale values (0-255) into a range between 0-1
def normalize(new_train_images):
    for i in range(len(new_train_images)):
        for j in range(len(new_train_images[i])):
            new_train_images[i,j] = new_train_images[i,j]/255
    return new_train_images

def one_hot_encoding(digit):

    vector_output = numpy.zeros(10,dtype=int)
    vector_output[digit] = 1 
    return vector_output
def ReLU(FHL):
    for i in range(1000):
        if FHL[i] <= 0:
            FHL[i] = 0
    return FHL
#Calculates the activation ouput
def activation(train_images_normalized,bias,weight):
    FHL = numpy.zeros(1000)
    for i in range(1000):
        FHL[i] = weight[i]@train_images_normalized[i]
        FHL = FHL + bias
    FHL = ReLU(FHL)
    return FHL

weight = numpy.random.normal(loc=0,scale=numpy.sqrt((2/784)),size=(1000,784))
bias = numpy.zeros((1000))
new_train_images = reshape(train_images)
train_images_normalized = normalize(new_train_images)
print(weight.shape,train_images_normalized.shape)
FHL = activation(train_images_normalized,bias,weight)
#vector_output = one_hot_encoding(train_labels[0])



