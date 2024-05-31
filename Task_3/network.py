import torch
import torch.nn as nn
from sklearn import preprocessing     

class Recurrent_CNN(nn.Module):
    def __init__(self,
                 num_classes: int,

                 Conv_Channels : int = 32, conv_kern : int = 7, conv_stride : int = 1, conv_padd : int = 3,
                 Conv_dropout : float = 0.25,

                 ResNet1_channels : int = 64, ResNet1_kern : int = 3, ResNet1_stride : int = 1, ResNet1_pad : int = 1,
                 ResNet2_channels : int = 128, ResNet2_kern : int = 3, ResNet2_stride : int = 1, ResNet2_pad : int = 1,
                 ResNet3_channels : int = 256, ResNet3_kern : int = 3, ResNet3_stride : int = 1, ResNet3_pad : int = 1,

                 rnn_size : int = 256, rnn_dropout : float = 0.25
                 ):
        """
        Initialize a RCNN model
        @param input_size: size of the input tensor
        @param num_classes: number of classes to predict
        
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
            nn.Dropout(Conv_dropout),

            nn.MaxPool2d(2)
        )

        # First ResNet Block
        self.ResNet_Module_1 = nn.Sequential(
            nn.Conv2d(Conv_Channels, ResNet1_channels, ResNet1_kern, ResNet1_stride, ResNet1_pad),
            nn.BatchNorm2d(ResNet1_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            nn.Conv2d(ResNet1_channels, ResNet1_channels, ResNet1_kern, ResNet1_stride, ResNet1_pad),
            nn.BatchNorm2d(ResNet1_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            # Includes Max Pooling at the end
            nn.MaxPool2d(2)
        )

        # Second ResNet Block
        self.ResNet_Module_2 = nn.Sequential(
            nn.Conv2d(ResNet1_channels, ResNet2_channels, ResNet2_kern, ResNet2_stride, ResNet2_pad),
            nn.BatchNorm2d(ResNet2_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            nn.Conv2d(ResNet2_channels, ResNet2_channels, ResNet2_kern, ResNet2_stride, ResNet2_pad),
            nn.BatchNorm2d(ResNet2_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            nn.Conv2d(ResNet2_channels, ResNet2_channels, ResNet2_kern, ResNet2_stride, ResNet2_pad),
            nn.BatchNorm2d(ResNet2_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            nn.Conv2d(ResNet2_channels, ResNet2_channels, ResNet2_kern, ResNet2_stride, ResNet2_pad),
            nn.BatchNorm2d(ResNet2_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            # Includes Max Pooling at the end
            nn.MaxPool2d(2)
        )

        # Third ResNet Block
        self.ResNet_Module_3 = nn.Sequential(
            nn.Conv2d(ResNet2_channels, ResNet3_channels, ResNet3_kern, ResNet3_stride, ResNet3_pad),
            nn.BatchNorm2d(ResNet3_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            nn.Conv2d(ResNet3_channels, ResNet3_channels, ResNet3_kern, ResNet3_stride, ResNet3_pad),
            nn.BatchNorm2d(ResNet3_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            nn.Conv2d(ResNet3_channels, ResNet3_channels, ResNet3_kern, ResNet3_stride, ResNet3_pad),
            nn.BatchNorm2d(ResNet3_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout),

            nn.Conv2d(ResNet3_channels, ResNet3_channels, ResNet3_kern, ResNet3_stride, ResNet3_pad),
            nn.BatchNorm2d(ResNet3_channels),
            nn.ReLU(),
            nn.Dropout(Conv_dropout)
            # No max pooling at the end
        )
        #Max Pooling for each column
        self.ColumnPooling_Module = nn.AdaptiveMaxPool2d((1, None))

        self.Recurrent_Module = nn.Sequential(
            nn.LSTM(input_size = ResNet3_channels, 
                    hidden_size = rnn_size, 
                    num_layers = 3, batch_first = True, dropout = rnn_dropout, bidirectional = True)
        )

        self.Output_Module = nn.Sequential(
            nn.Dropout(rnn_dropout),
            nn.Linear(rnn_size * 2, num_classes),
            nn.Softmax(dim = 1)
        )


        self.CTC_Block = nn.Conv1d(ResNet3_channels, num_classes, 3, 1, 1)

    
        return

    def forward(self, image : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        @param x: input tensor
        @return: output tensor
        """
        batch_size, channels, height, widht = image.size()
        x = self.Convolutional_Module(image)
        x = self.ResNet_Module_1(x)
        x = self.ResNet_Module_2(x)
        x = self.ResNet_Module_3(x)
        #(Batchsize, 256, 16, 128)
        x = self.ColumnPooling_Module(x)
        #(Batchsize, channels 256, height 1, width 128)

        # Forward pass through RNN modules
        output_rnn, state_rnn = self.Recurrent_Module(x.permute(0, 2, 3, 1).squeeze(1))
        output_rnn = self.Output_Module(output_rnn)

        # Forward output through ctc module
        output_ctc = self.CTC_Block(x.squeeze(2))

        return output_rnn, output_ctc

