import torch
import torch.nn as nn
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self, dim_b, dim_c, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_b + dim_c, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, joint):
        return self.net(joint)


class TissueDiscriminator(nn.Module):
    def __init__(self, latent_dim, output_dim, dropout=0.2, hidden_num=0):
        super().__init__()
        if hidden_num ==0:
            self.net = nn.Sequential(
                nn.Linear(latent_dim, output_dim)
            )
        else:
            layers = []
            # 输入层
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # 隐藏层
            for _ in range(hidden_num - 1):
                layers.append(nn.Linear(latent_dim, latent_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            # 输出层
            layers.append(nn.Linear(latent_dim, output_dim))
            
            self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class TissueDiscriminator2(nn.Module):
    def __init__(self, latent_dim, output_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, output_dim)
        )
    def forward(self, z):
        return self.net(z)
            


def adversarial_loss(z_bio, z_conf, discriminator):
    # 判别器学习区分真实独立样本和模型生成的联合分布
    z_joint = torch.cat([z_bio, z_conf], dim=1)
    z_shuffled = torch.cat([z_bio, z_conf[torch.randperm(z_conf.size(0))]], dim=1)
    
    real_labels = torch.ones(z_joint.size(0), 1).to(z_joint.device)
    fake_labels = torch.zeros(z_shuffled.size(0), 1).to(z_shuffled.device)
    
    d_real = discriminator(z_joint)
    d_fake = discriminator(z_shuffled.detach())
    
    d_loss = F.binary_cross_entropy(d_real, real_labels) + \
             F.binary_cross_entropy(d_fake, fake_labels)
    
    # 生成器损失：欺骗判别器认为联合分布是独立的
    g_loss = F.binary_cross_entropy(d_real, fake_labels)
    return d_loss, g_loss

