import torch
import torch.nn as nn
import torchvision.models as models

class Resnet18(nn.Module):
    def __init__(self,num_class,pretrained=False,pretrained_model='./pretrained/resnet18_msceleb.pth'):
        super(Resnet18,self).__init__()
        resnet=models.resnet18(pretrained=pretrained)
        if pretrained:
            try:
                checkpoint=torch.load(pretrained_model)
                resnet.load_state_dict(checkpoint['state_dict'],strict=True)
            except:
                print("There is not pretrained wights :{} ,Please check again!".format(pretrained_model))
        

        self.features=nn.Sequential(*list(resnet.children())[:-2])
        self.fc=nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(512,num_class),
            #nn.Sigmoid()
        )


    
    def forward(self,x):
        x=self.features(x)
        out=self.fc(x)
        return out