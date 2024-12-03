import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import csv

################## INITIALIZATION #######################

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

with open("F_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for x in reader:
        F_data += x


################## REPRESENTATION #######################
#create tensor for inputs
tensorList = [[cartPosition, cartVelocity, pendulumAngle, angularSpeed]]
for i in range(x_data.size()):
    temp = [x_data[i], dx_data[i], th_data[i], dth_data[i]]
    tensorList += temp
inputs = torch.tensor(tensorList, dType = torch.float32)
targetVals = torch.tensor(F_data, dType = torch.float32)

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
loss_function = nn.MSELoss()
optimization = optim.SGD(model.parameters(), lr = 0.01)

################## OPTIMIZATION #########################
#also known as epochs (idk why)
iterations = 100*101

for x in range(iterations):
    #clear gradients
    optimization.zero_grad()
    #forward pass
    outputs = cartModel(inputs)
    loss = loss_function(outputs, targetVals)
    #backwards pass/ back prop
    loss.backward()
    #update weights!
    optimization.step()

    #Show one out of every 50 values to test
    if(x + 1) % 50 == 0:
        print("Epoch [{x + 1}/{iterations}], Loss: {loss.item().4f}")

#Evaluation time!
#disabling gradient tracking, to save on much needed computing power
with torch.no_grad():
    #picking a random set of values from tensorList to test the neural network
    testValue = 100;
    testTensor = torch.tensor([tensorList[testValue]])
    #neural network's prediction of what the force should be
    forceOutput = cartModel(testTensor)
    print("Calculated force: ", forceOutput)
    
