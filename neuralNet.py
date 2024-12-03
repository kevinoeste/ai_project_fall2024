import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

################## INITIALIZATION #######################

#define directions
#tensors that only have a single value are scalars
forwards = 1
backwards = -1

#The four states that we need to control:
#X position of cart
#Velocity of cart
#angle of pendulum
#angular speed of pendulum

num_inputs = 4
#this is an arbitrary number
hidden_layer_size = 16
num_outputs = 1

################## REPRESENTATION #######################
#create tensor for inputs
inputs = torch.tensor([[cartPosition, cartVelocity, pendulumAngle, angularSpeed]])

class cartNN(nn.Module):
  def __init__(self, hiddenSize):
    super(cartNN, self).__init__()
    self.hiddenLayer1 = nn.Linear(4, hiddenSize)
    self.outputLayer = nn.Linear(hiddenSize, 1)
  def forward(self, f):
    #use ReLU on the first hidden layer
    f = F.relu(self.hiddenLayer1(x))
    f = self.outputLayer(f)
    return f
  



################## OPTIMIZATION #########################
