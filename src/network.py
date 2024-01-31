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


class Dynamics(nn.Module):
    '''Hidden state transition'''
    def __init__(self, num_channels=config.num_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(config.num_hidden + 2, num_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            ResidualBlock(num_channels),
            nn.Conv2d(num_channels, config.num_hidden, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(2),
            nn.ReLU(),

        )

    def forward(self, rp, a):
        h = torch.cat((rp, a), dim=1)
        h = self.conv(h)

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
        self.dynamics = Dynamics()
    
    def initial_inference(self, encoded_state):
        if encoded_state.ndim == 3:
            encoded_state = encoded_state.unsqueeze(0)
        hidden = self.representation(encoded_state)
        value, policy = self.prediction(hidden)
        return value, policy, hidden

    def recurrent_inference(self, hidden_state, encoded_action):
        if hidden_state.ndim == 3:
            hidden_state = hidden_state.unsqueeze(0)
        if encoded_action.ndim == 3:
            encoded_action = encoded_action.unsqueeze(0)
        # dynamics + prediction function
        hidden = self.dynamics(hidden_state, encoded_action)
        value, policy = self.prediction(hidden)
        return value, policy, hidden

    def forward(self):
        pass


def test():
    encoded_state = torch.randn(10, 3, 8, 8)
    encoded_action = torch.randn((10, 2, 8, 8))

    network = Network()
    with torch.no_grad():
        value, policy, hidden = network.initial_inference(encoded_state)
    print(f'value_ini:{value.shape}|policy_ini:{policy.shape}|hidden_ini:{hidden.shape}')

    with torch.no_grad():
        value, policy, hidden = network.recurrent_inference(hidden, encoded_action)
    print(f'value_rec:{value}|policy_rec:{policy.shape}|hidden_rec:{hidden.shape}')

if __name__ == "__main__":
    test()