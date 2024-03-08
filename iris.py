import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=17)
    
    scalar = StandardScaler()
    Xtrain= scalar.fit_transform(Xtrain)
    Xtest = scalar.transform(Xtest)

    Xtrain_tensor = torch.tensor(Xtrain, dtype=torch.float32)
    Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.long)
    ytest_tensor = torch.tensor(ytest, dtype=torch.long)
    
    return Xtrain_tensor, Xtest_tensor, ytrain_tensor, ytest_tensor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4,10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, loss_fn, optimizer, Xtrain, ytrain, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(Xtrain)
        loss = loss_fn(outputs, ytrain)
        loss.backward()
        optimizer.step()

        print('Epoch: %s | Loss: %s' % (epoch, loss.item()))

def evaluate_model(model, Xtest, ytest):
    with torch.no_grad():
        model.eval()
        outputs = model(Xtest)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(ytest.numpy(), predicted.numpy())
        print('Accuracy: %s' % (accuracy))

def main():
    Xtrain, Xtest, ytrain, ytest = load_and_preprocess_data()
    
    model = Model()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    train_model(model, loss_fn, optimizer, Xtrain, ytrain, 100)

    evaluate_model(model, Xtest, ytest)

if __name__ == '__main__':
    main()

