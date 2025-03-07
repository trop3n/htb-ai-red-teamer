# Save model
torch.save(model.state_dict(), 'model.pth')

# load model
model = CustomModel()
model = CustomModel()
model.load_state_dict(torch.load('model.pth'))
model.eval() # set the model to evaluation mode