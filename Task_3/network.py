import torch
import torch.nn as nn  
import numpy as np


class ResNetBlock(nn.Module):
    """Implementation of a ResNet block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bnskip = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.bnskip(self.skip(x))
        out = self.relu(out)
        return out


class Recurrent_CNN(nn.Module):
    def __init__(self,
                 num_classes: int,

                 Conv_Channels : int = 32, conv_kern : int = 7, conv_stride : int = 1, conv_padd : int = 3,
                 Conv_dropout : float = 0.0,

                 ResNet1_channels : int = 64, ResNet1_kern : int = 3, ResNet1_stride : int = 1, ResNet1_pad : int = 1,
                 ResNet2_channels : int = 128, ResNet2_kern : int = 3, ResNet2_stride : int = 1, ResNet2_pad : int = 1,
                 ResNet3_channels : int = 256, ResNet3_kern : int = 3, ResNet3_stride : int = 1, ResNet3_pad : int = 1,

                 rnn_size : int = 256, rnn_dropout : float = 0.2
                 ):
        """
        Initialize a RCNN model
        @param input_size: size of the input tensor
        @param num_classes: number of classes to predict
        """
        super(Recurrent_CNN, self).__init__()
        # Set up Convolutional Layers
        self.Convolutional_Module = nn.Sequential(
            nn.Conv2d(1, Conv_Channels, conv_kern, conv_stride, conv_padd),
            nn.BatchNorm2d(Conv_Channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            nn.MaxPool2d(2)
        )

        # First ResNet Block
        self.ResNet_Module_1 = nn.Sequential(
            ResNetBlock(Conv_Channels, ResNet1_channels),
            ResNetBlock(ResNet1_channels, ResNet1_channels),
            nn.MaxPool2d(2),
        )

        # Second ResNet Block
        self.ResNet_Module_2 = nn.Sequential(
            ResNetBlock(ResNet1_channels, ResNet2_channels),
            ResNetBlock(ResNet2_channels, ResNet2_channels),
            ResNetBlock(ResNet2_channels, ResNet2_channels),
            ResNetBlock(ResNet2_channels, ResNet2_channels),
            nn.MaxPool2d(2),
        )

        # Third ResNet Block
        self.ResNet_Module_3 = nn.Sequential(
            ResNetBlock(ResNet2_channels, ResNet3_channels),
            ResNetBlock(ResNet3_channels, ResNet3_channels),
            ResNetBlock(ResNet3_channels, ResNet3_channels),
            ResNetBlock(ResNet3_channels, ResNet3_channels),
            # No max pooling at the end!
        )
       
        self.ColumnPooling_Module = nn.AdaptiveMaxPool2d((1, None))

        # 1. Output Module: RNN branch
        self.Recurrent_Module = nn.Sequential(
            nn.LSTM(input_size = ResNet3_channels, 
                    hidden_size = rnn_size, 
                    num_layers = 3, batch_first = True, dropout = rnn_dropout, bidirectional = True)
        )
        # 2. Output Module: CTC branch
        self.Output_Module = nn.Sequential(
            nn.Dropout(rnn_dropout),
            nn.Linear(in_features = rnn_size * 2,out_features = num_classes)
        )


        self.CTC_Block = nn.Sequential(
            nn.Conv1d(in_channels = ResNet3_channels,out_channels = num_classes, kernel_size = 3,stride = 1, padding = 1)
        )

    
        return

    def forward(self, image : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        @param x: input tensor of size (Batchsize, num_channels = 1, height = 128, width = 1024)
        @return: output two tensor of shape (Batchsize, SeqLen = 128, NumClasses = 85 + 1) for RNN + CTC modules
        """
        x = self.Convolutional_Module(image)
        x = self.ResNet_Module_1(x)
        x = self.ResNet_Module_2(x)
        x = self.ResNet_Module_3(x)         #(Batchsize, 256, 16, 128)
        
        x = self.ColumnPooling_Module(x)    #(Batchsize, channels 256, height 1, width 128)
        
        # Forward through RNN Module
        output_rnn, state_rnn = self.Recurrent_Module(x.permute(0, 2, 3, 1).squeeze(1))
        output_rnn = self.Output_Module(output_rnn)

        # Forward output through CTC module
        output_ctc = self.CTC_Block(x.squeeze(2))

        return output_rnn, output_ctc.permute(0, 2, 1)



class AugmentAgentCNN(nn.Module):
    def __init__(
        self,
        n_points: int
    ):
        super(AugmentAgentCNN, self).__init__()
        self.n_points = n_points
        self.n_patches = (n_points//2)-1

        self.pool = nn.AvgPool2d(kernel_size=(2, 2))
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(3,3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 16, kernel_size=(3,3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(192, 8*(self.n_patches+1))

    def forward(
        self, state: np.ndarray
    ):
        x = self.pool(nn.functional.relu(self.conv1(state)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = nn.functional.relu(self.bn1(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = nn.functional.relu(self.bn2(self.conv5(x)))
        x = self.pool(nn.functional.relu(self.bn3(self.conv6(x))))
        x = x.view(-1, 192)
        x = self.fc1(x)
        x = x.view(-1, self.n_points, 2, 2)
        x = nn.functional.softmax(x,3)
        return x