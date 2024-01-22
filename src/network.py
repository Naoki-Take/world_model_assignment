import config
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            # nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, 1, 1),
            # nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class Representation(nn.Module):
    def __init__(self, num_channels=config.num_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(config.state_shape[0], num_channels, 3, 1, 1),
            # nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(num_channels, config.num_hidden, 3, 1, 1),
            nn.ReLU(),
        )
    
    def forward(self, encoded_state):
        h = self.conv1(encoded_state)
        h = self.conv2(h)
        h = self.conv3(h)

        return h


class Prediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_size = config.board_length**2

        self.value_head = nn.Sequential(

            nn.Conv2d(config.num_hidden, 2, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size*2, 1),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(config.num_hidden, 2, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.board_size*2, self.board_size*2),
            nn.ReLU(),
            nn.Linear(self.board_size*2, config.action_space_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        value = self.value_head(x)
        policy = self.policy_head(x)
        
        return value, policy

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.representation = Representation()
        self.prediction = Prediction()
    
    def initial_inference(self, encoded_state):
        if encoded_state.ndim == 3:
            encoded_state = encoded_state.unsqueeze(0)
        hidden = self.representation(encoded_state)
        value, policy = self.prediction(hidden)
        return value, policy, hidden

    def forward(self):
        pass
def test():
    encoded_state = torch.randn(config.state_shape)
    action = torch.randn((1, 2, 8, 8))

    network = Network()

    value, policy, hidden = network.initial_inference(encoded_state)
    print(f'value_ini:{value.shape}|policy_ini:{policy.shape}|hidden_ini:{hidden.shape}')

if __name__ == "__main__":
    test()