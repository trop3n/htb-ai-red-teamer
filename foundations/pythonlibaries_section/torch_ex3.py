def dynamic_rnn(inputs):
    hidden_state = torch.zeros(1, 5) # initial hidden state
    for i in range(inputs.shape[0]): # graph grows with each loop iteration
        hidden_state = torch.mm(inputs[i], weights) + hidden_state
    return hidden_state