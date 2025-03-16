import torch.optim as optim
import torch.nn as nn
import torch

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / len(target)

epochs = 10
num_batches = 100
for epoch in range(epochs):
    for batch in range(num_batches):
        x_batch, y_batch = get_batch(batch)
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch+1}/{num_batches}], Loss: {loss.item():.4f}')