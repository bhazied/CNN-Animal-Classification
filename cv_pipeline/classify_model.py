import torch
import  torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .convblock import ConvBlock
class ClassifyModel(nn.Module):
    def __init__(self, num_classes = 10):
        super(ClassifyModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = nn.Sequential(
            ConvBlock(3, 16), # MaxPool2d (2,2) 28 --> 14
            ConvBlock(16, 32) # MaxPool2d (2,2) 14 --> 7
        )
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32*7*7, 128), # 32 as output of previous layer, and image 7X7
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.model(x)
        return x

    def fit(self, train_loader: DataLoader, eval_loader: DataLoader, loss_function, optimizer, num_epochs:int = 5):
        self.to(self.device)
        train_losses = []
        eval_losses = []
        eval_accuracies = []
        for epoch in range(num_epochs):
           self.train()
           epoch_train_loss = self.run_train(train_loader = train_loader, loss_function = loss_function, optimizer = optimizer)
           train_losses.append(epoch_train_loss)
           self.eval()
           eval_loss, eval_accuracy = self.evaluate(eval_loader, loss_function)
           eval_losses.append(eval_loss)
           eval_accuracies.append(eval_accuracy)
           print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {eval_loss:.4f}, Val Accuracy: {eval_accuracy:.2f}%")
        return [train_losses, eval_losses, eval_accuracies]

    def run_train(self, train_loader: DataLoader, loss_function, optimizer):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def evaluate(self, eval_loader:DataLoader, loss_function):
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
           for inputs, labels in eval_loader:
               i = 1
               inputs = inputs.to(self.device)
               labels = labels.to(self.device)
               outputs = self.forward(inputs)
               loss = loss_function(outputs, labels)
               running_loss += loss.item() * inputs.size(0)
               print (f"loss for batch {i}:  { loss.item()}")
               _, prediected = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (prediected == labels).sum().item()
               i+=1
        print (f"running loss {running_loss} et lenght {len(eval_loader.dataset)}")
        eval_loss = running_loss / len(eval_loader.dataset)
        eval_accuracy = 100 * correct / total
        return eval_loss, eval_accuracy
           


