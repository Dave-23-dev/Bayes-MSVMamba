import torch
from Experiment.model.demo import DemoModel

def train_demo_model():
    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 1
    num_epochs = 5
    learning_rate = 0.001

    # Create a model instance
    model = DemoModel(input_size, hidden_size, output_size)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy data for demonstration
    inputs = torch.randn(100, input_size)
    targets = torch.randn(100, output_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')