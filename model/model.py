########## U-Net model ###############

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models



class UNet(nn.Module):
    """
    num_classes: number of output classes
    min_channels: minimum number of channels in conv layers
    max_channels: number of channels in the bottleneck block
    num_down_blocks: number of blocks which end with downsampling
    """
    def __init__(self, 
                 num_classes,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):
        
        super(UNet, self).__init__()
        
        self.num_classes = num_classes
        
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.maxpool =  nn.MaxPool2d(kernel_size = 2, stride = 2) 
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
 
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2)
        
        
        self.layer4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(), 
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(), 
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(), 
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.layer1 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(), 
                                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())
     
        
        self.output = nn.Conv2d(32, self.num_classes, kernel_size=3, padding=1)


    def forward(self, inputs):
        
        x = F.interpolate(inputs, (inputs.shape[-2] // 16 * 16, inputs.shape[-1] // 16 * 16))

        x1 = self.down1(x)
        x1_pool = self.maxpool(x1)
        
        x2 = self.down2(x1_pool)
        x2_pool = self.maxpool(x2)
        
        x3 = self.down3(x2_pool)
        x3_pool = self.maxpool(x3)
        
        x4 = self.down4(x3_pool)
        x4_pool = self.maxpool(x4)
      
        x5 = self.bottleneck(x4_pool)    

        
        y5_up = self.up4(x5)        
        y5 = torch.cat([y5_up, x4], dim=1)
        y5 = self.layer4(y5)

        
        y4_up = self.up3(y5)
        y4 = torch.cat([y4_up, x3], dim=1)
        y4 = self.layer3(y4)
        
        y3_up = self.up2(y4)
        y3 = torch.cat([y3_up, x2], dim=1)
        y3 = self.layer2(y3)
        
        y2_up = self.up1(y3)
        y2 = torch.cat([y2_up, x1], dim=1)
        y2 = self.layer1(y2)
            
        y1 = self.output(y2)
        
        logits = F.interpolate(y1, (inputs.shape[2], inputs.shape[3]))
        

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits
