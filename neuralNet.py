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

#define default states
cartPosition = 0
cartVelocity = 0
pendulumAngle = 0
angularSpeed = 0

#create tensor for inputs
inputs = torch.tensor([[cartPosition, cartVelocity, pendulumAngle, angularSpeed]])


################## REPRESENTATION #######################




################## OPTIMIZATION #########################
