import torch
import torch.nn as nn
from sklearn import preprocessing     

class Recurrent_CNN(nn.Module):
    def __init__(self,
                 num_classes: int,

                 Conv_Channels : int = 32, conv_kern : int = 7, conv_stride : int = 1, conv_padd : int = 1,

                 ResNet1_channels : int = 64, ResNet1_kern : int = 3, ResNet1_stride : int = 1, ResNet1_pad : int = 1,
                 ResNet2_channels : int = 128, ResNet2_kern : int = 3, ResNet2_stride : int = 1, ResNet2_pad : int = 1,
                 ResNet3_channels : int = 256, ResNet3_kern : int = 3, ResNet3_stride : int = 1, ResNet3_pad : int = 0,

                 rnn_size_1 : int = 256, rnn_dropout_1 : float = 0.25,
                 rnn_size_2 : int = 256, rnn_dropout_2 : float = 0.25,
                 rnn_size_3 : int = 256, rnn_dropout_3 : float = 0.25
                 ):
        """
        Initialize a RCNN model
        @param input_size: size of the input tensor
        @param num_classes: number of classes to predict
        @param optimizer: torch optimizer to use for training
        
        @param cnn_channels_1: number of output channels in the first convolutional layer
        @param cnn_kernel_1: size of the kernel in the first convolutional layer
        @param cnn_stride_1: stride of the first convolutional layer
        @param cnn_padding_1: padding of the first convolutional layer

        @param rnn_size_1: number of neurons of the first recurrent layer
        @param rnn_dropout_1: dropout rate of the first recurrent layer
        """
        super(Recurrent_CNN, self).__init__()
        # Set up Convolutional Layers
        self.Convolutional_Module = nn.Sequential(
            nn.Conv2d(1, Conv_Channels, conv_kern, conv_stride, conv_padd),
            nn.BatchNorm2d(Conv_Channels),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        # First ResNet Block
        self.ResNet_Module_1 = nn.Sequential(
            nn.Conv2d(Conv_Channels, ResNet1_channels, ResNet1_kern, ResNet1_stride, ResNet1_pad),
            nn.BatchNorm2d(ResNet1_channels),
            nn.ReLU(),

            nn.Conv2d(ResNet1_channels, ResNet1_channels, ResNet1_kern, ResNet1_stride, ResNet1_pad),
            nn.BatchNorm2d(ResNet1_channels),
            nn.ReLU(),

            # Includes Max Pooling at the end
            nn.MaxPool2d(2)
        )

        # Second ResNet Block
        self.ResNet_Module_2 = nn.Sequential(
            nn.Conv2d(ResNet1_channels, ResNet2_channels, ResNet2_kern, ResNet2_stride, ResNet2_pad),
            nn.BatchNorm2d(ResNet2_channels),
            nn.ReLU(),

            nn.Conv2d(ResNet2_channels, ResNet2_channels, ResNet2_kern, ResNet2_stride, ResNet2_pad),
            nn.BatchNorm2d(ResNet2_channels),
            nn.ReLU(),

            nn.Conv2d(ResNet2_channels, ResNet2_channels, ResNet2_kern, ResNet2_stride, ResNet2_pad),
            nn.BatchNorm2d(ResNet2_channels),
            nn.ReLU(),

            nn.Conv2d(ResNet2_channels, ResNet2_channels, ResNet2_kern, ResNet2_stride, ResNet2_pad),
            nn.BatchNorm2d(ResNet2_channels),
            nn.ReLU(),

            # Includes Max Pooling at the end
            nn.MaxPool2d(2)
        )

        # Third ResNet Block
        self.ResNet_Module_3 = nn.Sequential(
            nn.Conv2d(ResNet2_channels, ResNet3_channels, ResNet3_kern, ResNet3_stride, ResNet3_pad),
            nn.BatchNorm2d(ResNet3_channels),
            nn.ReLU(),

            nn.Conv2d(ResNet3_channels, ResNet3_channels, ResNet3_kern, ResNet3_stride, ResNet3_pad),
            nn.BatchNorm2d(ResNet3_channels),
            nn.ReLU(),

            nn.Conv2d(ResNet3_channels, ResNet3_channels, ResNet3_kern, ResNet3_stride, ResNet3_pad),
            nn.BatchNorm2d(ResNet3_channels),
            nn.ReLU(),

            nn.Conv2d(ResNet3_channels, ResNet3_channels, ResNet3_kern, ResNet3_stride, ResNet3_pad),
            nn.BatchNorm2d(ResNet3_channels),
            nn.ReLU()
            # No max pooling at the end
        )

        self.ColumnPooling_Module = nn.AdaptiveAvgPool2d((1, None))

        self.Recurrent_Module = nn.Sequential(
            nn.LSTM(ResNet3_channels, rnn_size_1, 1, batch_first = True, dropout = rnn_dropout_1, bidirectional = True),
            nn.LSTM(rnn_size_1*2, rnn_size_2, 1, batch_first = True, dropout = rnn_dropout_2, bidirectional = True),
            nn.LSTM(rnn_size_2*2, rnn_size_3, 1, batch_first = True, dropout = rnn_dropout_3, bidirectional = True),

            nn.LazyLinear(num_classes + 1)
        )

        self.CTC_Block = nn.Conv2d(ResNet3_channels, num_classes + 1, 3, 1, 1)

    
        return

    def forward(self, image : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        @param x: input tensor
        @return: output tensor
        """
        bs, c, h, w = image.size()

        x = self.Convolutional_Module(image)
        x = self.ResNet_Module_1(x)
        x = self.ResNet_Module_2(x)
        x = self.ResNet_Module_3(x)
        x = self.ColumnPooling_Module(x)
        x = x.squeeze(2).permute(0, 2, 1)
        print(x.shape)
        print(type(x))

        output_rnn, state_rnn = self.Recurrent_Module(x)
        output_ctc = self.CTC_Block(x)

        return output_rnn, output_ctc

