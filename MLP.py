import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn import MSELoss
from torch.utils.data import Subset
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader

preprocessed_data = genfromtxt('data/preprocessed_data.csv', delimiter=',', dtype = str)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  


x_indices = [3, 11]
y_indices = [5]

x = []
y = []

for row in preprocessed_data:
	temp_x_list = []
	temp_y_list = []
	for i in range(len(row)):
		if i in x_indices:
			temp_x_list.append(float(row[i]))
		if i in y_indices:
			temp_y_list.append(float(row[i]))
	x.append(temp_x_list)
	y.append(temp_y_list)

x_ray = np.array(x)
y_ray = np.array(y)

indices_removed = np.random.choice(np.arange(x_ray.shape[0]), 747, replace = False)
x_ray = np.delete(x_ray, indices_removed, 0)
y_ray = np.delete(y_ray, indices_removed, 0)

X_train, X_test, y_train, y_test = train_test_split(x_ray, y_ray, train_size = 692000)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

train_data = []
test_data = []
for i in range(len(X_train)):
   train_data.append([X_train[i], y_train[i]])

for i in range(len(X_test)):
   test_data.append([X_test[i], y_test[i]])

trainloader = torch.utils.data.DataLoader(train_data,batch_size=1000)
testloader = torch.utils.data.DataLoader(test_data,batch_size=1000)
i1, l1 = next(iter(trainloader))
print(i1.shape)
print(l1.shape)

### Model definition ###
n = 2

# First we define the trainable parameters A and b 
A = torch.randn((1, n), requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Then we define the prediction model
def model(x_input):
	return A.mm(x_input) + b


### Loss function definition ###

def loss(y_predicted, y_target):
	return ((y_predicted - y_target)**2).sum()/y_predicted.shape[1]


### Training the model ###

# Setup the optimizer object, so it optimizes a and b.
optimizer = optim.Adam([A, b], lr=0.1)

num_epochs = 20

# Main optimization loop
for epoch in range(num_epochs):
	running_loss = 0.0
	print("Epoch {}/{}".format(epoch+1, num_epochs))
	count = 0
	
	# Set the gradients to 0.
	for i, val in enumerate(trainloader):
		count += 1
		optimizer.zero_grad()

		# Compute the current predicted y's from x_dataset
		y_predicted = model(torch.transpose(val[0], 0, 1).type(torch.FloatTensor))
		# See how far off the prediction is
		current_loss = loss(y_predicted, val[1])
		# Compute the gradient of the loss with respect to A and b.
		current_loss.backward()
		# Update A and b accordingly.
		optimizer.step()
		#print(f"Epoch = {epoch}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")
		running_loss += current_loss

	print("Epoch {}/{}. Running_Loss: {}".format(epoch+1, num_epochs, running_loss/count))