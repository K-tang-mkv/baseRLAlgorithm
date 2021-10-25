from torch import functional as F
import torch.nn as nn
import torch 


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # define struct of each layers' convolution
        # params: input_channels=3, output_channels=16(decided by the number of kernels)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # the number of features as input for the fully connected layer is decided by 
        # the output of convolution layers, actually depends on the image initial size,
        # kernel_size, stride and padding
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        # width of output of convolution layers
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # height of output of convolution layers
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        # width * height * channels
        linear_input_size = convw * convh * 32
        # fully connected layer
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = x.to(device)
        # the first convolution layer
        x = F.relu(self.bn1(self.conv1(x)))
        # the seconde convolution layer
        x = F.relu(self.bn2(self.conv2(x)))
        # the third convolution layer
        x = F.relu(self.bn3(self.conv3(x)))

        # return the output of final fully connected layer
        return self.head(x.view(x.size(0), -1))





        