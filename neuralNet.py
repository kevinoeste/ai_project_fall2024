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
#print every test_num values during the training phase
test_num = 100

#import test data
dth_data = []
th_data = []
x_data = []
dx_data = []
F_data = []

with open("dth_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for row in reader:
        for value in row:
            #print(value)
            dth_data.append(float(value))
#print(dth_data)

with open("th_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for row in reader:
        for value in row:
            th_data.append(float(value))
      
with open("dx_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for row in reader:
        for value in row:
            dx_data.append(float(value))

with open("x_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for row in reader:
        for value in row:
            x_data.append(float(value))

with open("F_data.csv", "r") as f:
    reader = csv.reader(f, delimiter = ",")
    for row in reader:
        for value in row:
            F_data.append(float(value))


################## REPRESENTATION #######################
#create tensor for inputs
tensorList = []
for i in range(len(x_data)):
    temp = [x_data[i], dx_data[i], th_data[i], dth_data[i]]
    tensorList.append(temp)
#print(tensorList)
inputs = torch.tensor(tensorList, dtype = torch.float32)
targetVals = torch.tensor(F_data, dtype = torch.float32)

class cartNN(nn.Module):
  def __init__(self, hiddenSize):
    super(cartNN, self).__init__()
    self.hiddenLayer1 = nn.Linear(4, hiddenSize)
    self.outputLayer = nn.Linear(hiddenSize, 1)
  def forward(self, f):
    #use ReLU on the first hidden layer
    f = F.relu(self.hiddenLayer1(f))
    f = self.outputLayer(f)
    return f
  
#instantiation
cartModel = cartNN(hidden_layer_size)

#create loss function and optimization function
#use mean squared error for regression
loss_function = nn.MSELoss()
optimization = optim.SGD(cartModel.parameters(), lr = 0.01)

################## OPTIMIZATION #########################
#also known as epochs
epochs = 100*101

#neural network training arc
for x in range(epochs):
    #clear gradients
    optimization.zero_grad()
    #forward pass
    outputs = cartModel(inputs)
    loss = loss_function(outputs, targetVals)
    #backwards pass/ back prop
    loss.backward()
    #update weights!
    optimization.step()

    #Show one out of every test_num values to test
    if(x + 1) % test_num == 0:
        print("Epoch [", (x + 1)/(epochs), "], Loss: ", loss.item())

#Evaluation time!
#disabling gradient tracking, to save on much needed computing power
with torch.no_grad():
    #picking a random set of values from tensorList to test the neural network
    testValue = 100;
    testTensor = torch.tensor([tensorList[testValue]], dtype = torch.float32)
    #neural network's prediction of what the force should be
    forceOutput = cartModel(testTensor)
    print("Calculated force: ", forceOutput)
    
