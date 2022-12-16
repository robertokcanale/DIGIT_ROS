from torchvision import models
from torch import nn
import torch

class Resnt18Rnn(nn.Module):
    def __init__(self, params_model):
        super(Resnt18Rnn, self).__init__()
        num_classes = params_model["num_classes"]
        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        
        baseModel = models.resnet18(pretrained=pretrained)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features*2, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)
    def forward(self, left, right):
        b_z, ts, c, h, w = left.shape
        ii = 0
        y_left = self.baseModel((left[:,ii]))
        y_right = self.baseModel((right[:,ii]))
        y = torch.cat([y_left, y_right], dim=1)
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, ts):
            y_left = self.baseModel((left[:,ii]))
            y_right = self.baseModel((right[:,ii]))
            y = torch.cat([y_left, y_right], dim=1)
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x    

def create_resnet18rnn(num_classes = 3):

    h, w =80, 60
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    params_model={
        "num_classes": num_classes,
        "dr_rate": 0.1,
        "pretrained" : True,
        "rnn_num_layers": 1,
        "rnn_hidden_size": 100,}
    model = Resnt18Rnn(params_model)  
    return model      

