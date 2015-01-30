"""
Python

Created by Zhixuan Wang  01/29/2015 14:11

this script uses the method provided to train and predict

"""


# Libraries
import mnist_loader as loader
import network as nw

data = loader.load_data_wrapper()
training_data = data[0]
validation_data = data[1]
test_data = data[2]

net = nw.Network([784, 100, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)