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
  def __init__(self):
    super().__init__()
    #define default states
    self.cartPosition = 0
    self.cartVelocity = 0
    self.pendulumAngle = 0
    self.angularSpeed = 0
    self.force = 0



################## OPTIMIZATION #########################
