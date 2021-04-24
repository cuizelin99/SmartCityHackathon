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
import torch.nn.functional as F

preprocessed_data = genfromtxt('data/preprocessed_data.csv', delimiter=',', dtype = str)
predict_raw_data = genfromtxt('data/test_kwh.csv', delimiter=',', dtype = str)

predict_data = []

skip_head = True

for row in predict_raw_data:
	if skip_head:
		skip_head = False
		continue
	temp_list = []
	temp_list.append(float(row[1]))
	temp_list.append(float(row[10]))
	temp_list.append(float(row[14]))
	# temp_list.append(float(row[16]))
	# temp_list.append(float(row[18]))
	# temp_list.append(float(row[20]))
	# temp_list.append(float(row[22]))
	# temp_list.append(float(row[24]))
	# temp_list.append(float(row[26]))
	# temp_list.append(float(row[27]))
	# temp_list.append(float(row[29]))
	predict_data.append(temp_list)


print("Predict Data length: ")
print(len(predict_data))

predict_data = np.array(predict_data)

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)  


x_indices = [7, 11, 12]
y_indices = [5]

num_inputs = len(x_indices)
num_outputs = len(y_indices)

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


# model definition
class MLP(Module):
	# define model elements
	def __init__(self, n_inputs, n_outputs):
		super(MLP, self).__init__()
		# input to first hidden layer
		self.hidden1 = Linear(n_inputs, n_inputs)
		kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
		self.act1 = ReLU()
		# second hidden layer
		self.hidden2 = Linear(n_inputs, n_inputs)
		kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
		self.act2 = ReLU()
		# third hidden layer and output
		self.hidden3 = Linear(n_inputs, n_outputs)
		xavier_uniform_(self.hidden3.weight)
		self.act3 = Sigmoid()
 
	# forward propagate input
	def forward(self, X):
		# input to first hidden layer
		X = self.hidden1(X)
		X = self.act1(X)
		 # second hidden layer
		X = self.hidden2(X)
		X = self.act2(X)
		# third hidden layer and output
		X = self.hidden3(X)
		X = self.act3(X)
		return X

### Loss function definition ###

#MSE loss
def mse(t1, t2):
	diff = t1 - t2
	return torch.sum(diff*diff)/diff.numel()


### Training the model ###

# Setup the optimizer object, so it optimizes a and b.
model = MLP(num_inputs, num_outputs)
optimizer = optim.SGD(model.parameters(), lr=1e-5)
loss_fn = F.mse_loss

num_epochs = 10

# Main optimization loop
for epoch in range(num_epochs):
	training_loss = 0.0
	print("Epoch {}/{}".format(epoch+1, num_epochs))
	count = 0
	
	# Set the gradients to 0.
	for i, val in enumerate(trainloader):
		count += 1
		# Compute the current predicted y's from x_dataset
		y_predicted = model(val[0].type(torch.FloatTensor))
		# See how far off the prediction is
		current_loss = loss_fn(y_predicted, val[1].type(torch.FloatTensor))
		# Compute the gradient of the loss with respect to A and b.
		current_loss.backward()
		# Update A and b accordingly.
		optimizer.step()
		optimizer.zero_grad()
		#print(f"Epoch = {epoch}, loss = {current_loss}, A = {A.detach().numpy()}, b = {b.item()}")
		training_loss += current_loss

	print("Epoch {}/{}. Training Loss: {}".format(epoch+1, num_epochs, training_loss/count))

	count = 0
	validation_loss = 0
	for i, val in enumerate(testloader):
		count += 1
		# Compute the current predicted y's from x_dataset
		y_predicted = model(val[0].type(torch.FloatTensor))
		# See how far off the prediction is
		current_loss = loss_fn(y_predicted, val[1].type(torch.FloatTensor))
		validation_loss += current_loss

	print("Epoch {}/{}. Validation Loss: {}".format(epoch+1, num_epochs, validation_loss/count))

predict_tensor = torch.from_numpy(predict_data).type(torch.FloatTensor)
yhat = model(predict_tensor)
yhat = yhat.detach().numpy()

np.savetxt('predictions.kwh', yhat, delimiter=',') 