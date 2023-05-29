import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Distangler(nn.Module):
    def __init__(self,input_channel=79,output_channel_1=128,output_channel_2=64):
        super(Distangler,self).__init__()
        # shared fully connect layers 
        self.activ = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_channel,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,256)
        # branch 1 to feature parameters for other features
        self.branch1 = nn.Linear(256,output_channel_1)
        # branch 2 to feature parameters for mouths
        self.branch2 = nn.Linear(256,output_channel_2)
    
    def forward(self,x):
        x = self.fc1(x)
        # x = F.normalize(x)
        x = self.activ(x)
        x = self.fc2(x)
        # x = F.normalize(x)
        x = self.activ(x)
        x = self.fc3(x)
        # x = F.normalize(x)
        x = self.activ(x)
        out1 = self.branch1(x)
        # out1 = F.normalize(out1)
        out2 = self.branch2(x)
        # out2 = F.normalize(out2)
        return out1,out2

class Concatenater(nn.Module):
    def __init__(self,input_channel_1=128,input_channel_2=64,output_channel=79):
        super(Concatenater,self).__init__()
        self.channel = input_channel_1 + input_channel_2
        self.activ = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channel,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,output_channel)
        self.output_activ = nn.Tanh()

    def forward(self,x1,x2):
        # concanate the two vectors
        x = torch.cat((x1,x2),dim=1)
        out = self.fc1(x)
        # out = F.normalize(out)
        out = self.activ(out)
        out = self.fc2(out)
        # out = F.normalize(out)
        out = self.activ(out)
        out = self.fc3(out)
        # out = self.output_activ(out)
        return out

class MouthExp2KptsNet(nn.Module):
    def __init__(self, input_dims=32, hidden_dims=64, num_hidden_layers=0, output_dims=20 * 2):
        super(MouthExp2KptsNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(inplace=True),
        )

        hidden_layers = []
        if num_hidden_layers > 0:
            for i in range(num_hidden_layers):
                hidden_layers_add = [
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.BatchNorm1d(hidden_dims),
                    nn.ReLU(inplace=True),
                ]
                hidden_layers.extend(hidden_layers_add)
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims, output_dims)
        )
    
    def forward(self, input):
        output = self.input_layer(input)
        # for layer in self.hidden_layers:
        output = self.hidden_layers(output)
        output = self.output_layer(output)

        return output

        

#class ExpressionNet(nn.Module):
    #def __init__(self,input_channel=79,output_channel=79):
       # super(ExpressionNet,self).__init__()
       # self.distangle = Distangler(input_channel)
       # self.concatenate = Concatenater(output_channel=output_channel)
        #self.mouth = MouthNet()
        #self.gan = GAN_net()
    #def forward(self,x):
       # out1,out2 = self.distangle(x)
        #out1 = self.mouth(out1)
        #out2 = self.gan(out2)
       # out = self.concatenate(out1,out2)
       # out += x
       # return out
