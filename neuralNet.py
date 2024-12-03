import torch
from torch import nn
from torch.utils.data import DataLoader

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





################## REPRESENTATION #######################
#create tensor for inputs
inputs = torch.tensor([[cartPosition, cartVelocity, pendulumAngle, angularSpeed]])

class cartNN(nn.Module):
  def __init__(self, hiddenSize):
    super(cartNN, self).__init__()
    #define default states
    #not sure if I'm gonna need these or not, nothing I see online has variables like this in the default constructor
    #self.cartPosition = 0
    #self.cartVelocity = 0
    #self.pendulumAngle = 0
    #self.angularSpeed = 0
    #self.force = 0
    self.hiddenLayer1 = nn.Linear(4, hiddenSize)
    self.outputLayer = nn.Linear(hiddenSize, 1)
  def forward(self, f):
    f = self.flatten(f)
    logits = self.linear_relu_stack(f)
    return logits



################## OPTIMIZATION #########################
