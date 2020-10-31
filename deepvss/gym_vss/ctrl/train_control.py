import torch
import torch.nn as nn
import numpy as np
import argparse
from ctrl.ctrl_model import ModelLSTM as CTRL_MODEL

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="train.csv", help="Tab separated train file")
parser.add_argument("--epocs", default=10000, help="training epocs")
parser.add_argument("--lr", default=0.0001, help="Learning rate")
parser.add_argument("--load", default=None, help="Load pretrained model")
parser.add_argument("--b", default=32, help="batch size, default 32")
args = parser.parse_args()

# Hyper-parameters
num_epochs = int(args.epocs)
learning_rate = float(args.lr)
file = args.file

data = np.genfromtxt(file, delimiter='\t', dtype=np.float32)

np.random.shuffle(data)




train_size = int(data.shape[0]*0.7)

print(data.shape)
print(train_size)

x_train = data[0:train_size, 0:4]
y_train = data[0:train_size, 4:6]

x_test = data[train_size:data.shape[0], 0:4]
y_test = data[train_size:data.shape[0], 4:6]

print("Data:%d, train:%d, test:%d" % (data.shape[0], x_train.shape[0], x_test.shape[0]))

input_size = x_train.shape[1]
output_size = y_train.shape[1]

# Linear regression model
model = CTRL_MODEL(input_size, 2*input_size, output_size)
print("%d->%d->%d"%(input_size, 2*input_size, output_size))

if args.load is not None:
    model.load_state_dict(torch.load(args.load))

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Convert numpy arrays to torch tensors
train_inputs = torch.from_numpy(x_train)
train_targets = torch.from_numpy(y_train)
test_inputs = torch.from_numpy(x_test)
test_targets = torch.from_numpy(y_test)

# Train the model
for epoch in range(num_epochs):

    optimizer.zero_grad()

    b_indexes = np.random.randint(0, train_inputs.shape[0], int(args.b))
    batch_inputs = train_inputs[b_indexes]
    batch_targets = train_targets[b_indexes]

    # Forward pass
    outputs = model(batch_inputs)
    loss = nn.MSELoss()(outputs, batch_targets)

    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 500 == 0:
        outputs = model(test_inputs)
        test_loss = nn.MSELoss()(outputs, test_targets)
        print ('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item(), test_loss))

# Plot the graph
predicted = model(torch.from_numpy(x_test)).detach().numpy()

vr_error = y_test[:, 0] - predicted[:, 0]
vl_error = y_test[:, 1] - predicted[:, 1]

sum_error_vr = 0
sum_error_vl = 0
for i in range(0,len(vr_error)):
    #print("l:%.2f - %.2f = %.2f"%(y_test[i,0], predicted[i,0], lin_error[i]))
    #print("r:%.2f - %.2f = %.2f" % (y_test[i, 1], predicted[i, 1], ang_error[i]))
    sum_error_vr += abs(vr_error[i])
    sum_error_vl += abs(vl_error[i])

print("vr_error:%.2f, vl_error:%.2f"%(sum_error_vr/len(vr_error), sum_error_vl/len(vl_error)))

#plt.plot(y_train[:,0:1], predicted[:,0:1], 'ro', label='Lin error')
#plt.legend()
#plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'ctrl_model.cpth')
