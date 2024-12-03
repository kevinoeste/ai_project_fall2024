import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import csv

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
cartPosition = cartVelocity = pendulumAngle = angularSpeed = 0

#this is an arbitrary number, can be tweaked
hidden_layer_size = 16

#import test data
dth_data = []
th_data = []
x_data = []
dx_data = []
#putting this here just in case, do not think that we need this (at least for the inputs)
F_data = []

with open("dth_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for x in reader:
        dth_data += x
#print(dth_data)

with open("th_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for x in reader:
        th_data += x
      
with open("dx_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for x in reader:
        dx_data += x

with open("x_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for x in reader:
        x_data += x


################## REPRESENTATION #######################
#create tensor for inputs
inputs = torch.tensor([[cartPosition, cartVelocity, pendulumAngle, angularSpeed]], dType = torch.float32)

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
  
#instantiation
cartModel = cartNN(hiddenLayerSize)

#create loss function and optimization function
#use mean squared error for regression
loss = nn.MSELoss()
optimization = optim.SGD(model.parameters(), lr = 0.01)


################## OPTIMIZATION #########################
#also known as epochs
iterations = 100*101

for x in range(iterations):
  optimization.zero_grad()
  outputs = model(
