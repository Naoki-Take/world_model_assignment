import config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VAE(nn.Module):
    def __init__(self,latent_variable_size, is_cuda=False):
        super().__init__()
        self.is_cuda = is_cuda
        #Encoder
        self.encoder_block = nn.Sequential(
            nn.Conv2d(config.state_shape[0], 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, config.num_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(config.num_hidden),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(config.board_length*config.board_length*config.num_hidden, latent_variable_size)
        self.fc2 = nn.Linear(config.board_length*config.board_length*config.num_hidden, latent_variable_size)
        self.relu = nn.ReLU(inplace=True)
        #Decoder
        self.fc3 = nn.Linear(latent_variable_size, config.board_length*config.board_length*config.num_hidden)
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(config.num_hidden,8, kernel_size=3, stride =1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8,config.state_shape[0], kernel_size=3, stride =1, padding=1),
            nn.Sigmoid()

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def encode(self, x):
        h = self.encoder_block(x)
        h = h.view(h.size(0), -1)
        #print((x>0.000).sum())
        z_mean = self.fc1(h)
        z_logvar  = self.fc2(h)
        return z_mean, z_logvar

    def reparametrize(self, mu, logvar, device):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(device)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = self.fc3(z)
        h = self.relu(h)
        #print(h.size())
        #print((h>0.000).sum())
        h = h.view(-1,config.num_hidden,config.board_length,config.board_length)
        x = self.decoder_block(h)

        return x

    def get_latent_var(self, x, device):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar, device)
        return z

    def forward(self, x, device):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar, device)
        res = self.decode(z)
        return res, mu, logvar



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

        self.vae = VAE(128) #Representation()
        self.prediction = Prediction()
        self.dynamics = Dynamics()

    def initial_inference(self, encoded_state, device):
        if encoded_state.ndim == 3:
            encoded_state = encoded_state.unsqueeze(0)
        hidden = self.vae.get_latent_var(encoded_state, device)
        hidden = hidden.view(hidden.size(0), config.num_hidden, config.board_length, config.board_length)
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