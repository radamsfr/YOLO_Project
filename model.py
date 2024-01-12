import torch
import torch.nn as nn

# CNN architecture used in YOLO V1
architecture_config = [
    # Tuples (kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    'pool',
    (3, 192, 1, 1),
    'pool',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'pool',
    # List [last int = num_repeats]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'pool',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# A CNN model that will be reused to build the Yolo model
class CNNblock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNblock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyReLU = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.leakyReLU(x)
        return output
    
# Entire model, including Convolution, Maxpool and FCLs
class Yolo(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolo, self).__init__()
        self.architechture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architechture)
        self.fcs = self.create_fcs(**kwargs)
        
    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        output = self.fcs(x)
        return output
    
    def create_conv_layers(self, architechture):
        layers = []
        in_channels = self.in_channels
        
        for x in architechture:
            if type(x) == tuple:
                layers.append(CNNblock(in_channels=in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3]))
                in_channels = x[1]
            elif type(x) == str:
                layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                reps = x[2]
                
                for i in range(reps):
                    layers.append(CNNblock(in_channels=in_channels, out_channels=conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3]))
                    layers.append(CNNblock(in_channels=conv1[1], out_channels=conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3]))
                    in_channels = conv2[1]
                  
        # print(layers)
        # nn.Sequential takes the input and makes each a layer in a NN.
        # *layers 'unpacks' the layers array
        return nn.Sequential(*layers)
    
    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            # Inputs from CNN into linear layer
            nn.Flatten(),
            nn.Linear(1024 * S * S, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(512, S * S * (C + B * 5))
            # Outputs in classes + bounding boxes
            )
  
      
def test(S = 7, B = 2, C = 90):
    model = Yolo(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)

# test()