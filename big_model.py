## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        #self.fc1 = nn.Linear(256*11*11, 12800)
        #self.fc1_drop = nn.Dropout(p=0.4)
        
        #self.fc2 = nn.Linear(12800, 6000)
        self.fc1 = nn.Linear(256*11*11, 6000)
        self.fc1_drop = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(6000, 1000)
        self.fc2_drop = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(1000, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #print ('input size',x.size())
        conv1_x = F.relu(self.conv1(x))
        #print ('conv1_x size',conv1_x.size())
        pool1_x = self.pool1(conv1_x)
        #print ('pool1_x size',pool1_x.size())
        
        conv2_x = F.relu(self.conv2(pool1_x))
        #print ('conv2_x size',conv2_x.size())
        pool2_x = self.pool2(conv2_x)
        #print ('pool2_x size',pool2_x.size())
        
        conv3_x = F.relu(self.conv3(pool2_x))
        #print ('conv3_x size',conv3_x.size())
        pool3_x = self.pool3(conv3_x)
        #print ('pool3_x size',pool3_x.size())
        
        conv4_x = F.relu(self.conv4(pool3_x))
        #print ('conv4_x size',conv4_x.size())
        pool4_x = self.pool4(conv4_x)
        #print ('pool4_x size',pool4_x.size())
        
        flat_x = pool4_x.view(pool4_x.size(0), -1)
        #print ('flat_x size',flat_x.size())
        
        fc1_x = F.relu(self.fc1(flat_x))
        fc1_drop_x = self.fc1_drop(fc1_x)
        #print ('fc1_drop_x size',fc1_drop_x.size())
        
        fc2_x = F.relu(self.fc2(fc1_drop_x))
        fc2_drop_x = self.fc2_drop(fc2_x)
        #print ('fc2_drop_x size',fc2_drop_x.size())
        
        #fc3_x = F.relu(self.fc3(fc2_drop_x))
        #fc3_drop_x = self.fc3_drop(fc3_x)
        #print ('fc3_drop_x size',fc3_drop_x.size())
        
        fc3_x = self.fc3(fc2_drop_x)
        #print ('fc4_x size',fc4_x.size())
        
        return fc3_x
        
