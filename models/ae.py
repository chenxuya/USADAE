
import torch
import torch.nn as nn

class Confounder(nn.Module):
    def __init__(self, input_dim,latent_dim,hidden_dims=[128,64], dorp_out=0.2):
        super(Confounder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], latent_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(dorp_out)
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        z = self.bn3(self.fc3(x))
        return z

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256]):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], latent_dim)  # 总潜在维度是 latent_dim
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        z = self.fc3(x)
        z = self.bn3(z)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, confounder_dim, hidden_dims=[256,512]):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim+confounder_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

    def forward(self, z):
        z = torch.relu(self.bn1(self.fc1(z)))
        z = torch.relu(self.bn2(self.fc2(z)))
        x_hat = self.fc3(z)
        return x_hat

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim, confounder_dim,drop_out=0.2, hidden_dims=[512, 256]):
        super(AE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dims)
        self.decoder = Decoder(latent_dim, input_dim, confounder_dim, list(reversed(hidden_dims)))
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

class TwinAE(nn.Module):
    def __init__(self, input_dim, latent_dim, confounder_dim, drop_out=0.2, hidden_dims=[512, 256]):
        
        super(TwinAE, self).__init__()
        self.ae = AE(input_dim, latent_dim, confounder_dim, drop_out, hidden_dims)
        self.confounder = Confounder(input_dim, confounder_dim, [int(i / 2) for i in hidden_dims])
        
    def forward(self, x):
        bio = self.ae.encoder(x)
        bio = self.ae.drop_out(bio)
        confounder = self.confounder(x)
        h = torch.cat([bio, confounder], dim=1)
        h = self.ae.drop_out(h)
        x_hat = self.ae.decoder(h)
        return x_hat, bio,confounder

