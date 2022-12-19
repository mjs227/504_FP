import torch
import json
import random
import math
import torch.nn as nn


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch_dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float32


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28, 1)

    def forward(self, x):
        x_ = torch.einsum('ki,kj->kij', x, x)
        x_ = torch.reshape(x_, (x_.shape[0], x_.shape[1] ** 2))
        x_ = torch.cat((x, x_[:, 7:8], x_[:, 14:16], x_[:, 21:24], x_[:, 28:32], x_[:, 35:40], x_[:, 42:48]), dim=1)

        return torch.flatten(torch.sigmoid(self.fc1(x_)))


class NNModel:
    def __init__(self):
        self.net = NeuralNetwork()
        self.loss_fn = nn.BCELoss(weight=weight_tensor)

    def train(self, x, y, epochs=100, lr=1e-3, weight_decay=0, print_rate=10):
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        for i in range(epochs):
            h = self.net(x)
            loss = self.loss_fn(h, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = loss.item()

            if i == epochs - 1:
                print('Epoch: ' + str(i + 1))
                print('Loss: ' + str(total_loss))

                return h.detach().cpu()
            elif i % print_rate == print_rate - 1:
                print('Epoch: ' + str(i + 1))
                print('Loss: ' + str(total_loss))

    def test(self, x, y):
        self.net.eval()
        y_ = self.net(x)
        loss = self.loss_fn(y_, y).item()

        return loss, y_.detach().cpu()


with open('vector_list_bin_ref_per.json', 'r') as f:
    file = json.load(f)

indices = list(range(len(file['x'])))
test_indices = random.sample(indices, math.floor(len(file['x']) * 0.2))
train_indices = [x for x in indices if x not in test_indices]
x_test = torch.tensor([file['x'][i] for i in test_indices])
y_test = torch.tensor([file['y'][i] for i in test_indices])
x_train = torch.tensor([file['x'][i] for i in train_indices])
y_train = torch.tensor([file['y'][i] for i in train_indices])

train_sum = torch.sum(y_train, dim=0)
test_sum = torch.sum(y_test, dim=0)
class_vals = train_sum + test_sum
total_vals = torch.sum(class_vals).item()
weight_tensor = torch.tensor([1 - (class_vals[i].item() / total_vals) for i in range(4)])

weight_tensor.to(DEVICE)
x_train.to(DEVICE)
y_train.to(DEVICE)
x_test.to(DEVICE)
y_test.to(DEVICE)
weight_tensor = weight_tensor.type(torch_dtype)
x_test = x_test.type(torch_dtype)
y_test = y_test.type(torch_dtype)
x_train = x_train.type(torch_dtype)
y_train = y_train.type(torch_dtype)
model = NNModel()
model.nets.to(DEVICE)

train_eval_t = model.train(x_train, y_train, lr=0.3, weight_decay=0.0, epochs=5000, print_rate=100)
train_eval_t_ = torch.flatten(torch.argmax(train_eval_t, dim=1))
y_train_ = torch.flatten(torch.argmax(y_train.detach().cpu(), dim=1))
acc = (torch.sum(train_eval_t_ == y_train_).item() / y_train_.shape[0]) * 100

print('\n\nFinal train accuracy: ' + str(round(acc, 3)) + '%')

test_loss, test_eval_t = model.test(x_test, y_test)
test_eval_t_ = torch.flatten(torch.argmax(test_eval_t, dim=1))
y_test_ = torch.flatten(torch.argmax(y_test.detach().cpu(), dim=1))
acc = (torch.sum(test_eval_t_ == y_test_).item() / y_test_.shape[0]) * 100

print('Test loss: ' + str(test_loss))
print('Test accuracy: ' + str(round(acc, 3)) + '%')
print(y_test_.tolist())
print(test_eval_t_.tolist())
print(set(test_eval_t_.tolist()))
