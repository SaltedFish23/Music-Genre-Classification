import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import random
import pandas
import numpy
import matplotlib.pyplot as plt
from IPython import display
import visdom


class classfier(nn.Module):
    def __init__(self):
        
    
    def forward(self,X):
        


class accumulator:
    def __init__(self,n):
        self.data = [0.0]*n
        
    def add(self,*args):
        self.data = [a + float(b) for a,b in zip(self.data,args)]
        
    def reset(self):
        self.data = [0.0]*len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]

class visualize(object):
    def __init__(self,env='default',**kwargs):
        self.vis = visdom.Visdom(env=env,**kwargs)
        self.index = {}
        self.vis.line([0.],[0.],win='classifier',opts=dict(title = 'classifier'))
    
    def paint(self,train_loss,test_loss,accuracy,epochs):
        self.vis.line([[train_loss,test_loss,accuracy]],[epochs],win='train',update='append')

def data_iter():
    

def loss():
    

def optimize():
    

def train_epoch():


def train():
    

def updater():
    

