# spiking_mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
import snntorch.functional as SF
import snntorch.surrogate as surrogate
import matplotlib.pyplot as plt

# --------------------------
# 1. Hyperparameters
# --------------------------
batch_size = 128
num_epochs = 2
num_steps = 50   # time steps (spiking duration)
lr = 1e-3        # learning rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 2. Data Loading (MNIST)
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --------------------------
# 3. Define Spiking Neural Network
# --------------------------
beta = 0.9  # membrane decay

class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 1000)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(1000, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()  # initialize hidden states
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x.view(batch_size, -1))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)

# --------------------------
# 4. Training Loop
# --------------------------
net = SNN().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

def train():
    net.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            spk_rec, _ = net(data)
            loss = loss_fn(spk_rec, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += SF.accuracy_rate(spk_rec, targets)

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/len(train_loader):.4f}")

# --------------------------
# 5. Test Loop
# --------------------------
def test():
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spk_rec, _ = net(data)
            correct += SF.accuracy_rate(spk_rec, targets)
    print(f"Test Accuracy: {correct/len(test_loader):.4f}")

# --------------------------
# Run Training + Testing
# --------------------------
if __name__ == "__main__":
    train()
    test()
