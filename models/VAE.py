import torch
from torch import nn
import torch.nn.functional as F
from utils import init_weights, get_activation

class VAE(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.use_user_bias = args.get('use_user_bias')
        self.use_movie_bias = args.get('use_movie_bias')
        self.num_users = args.get('num_users')
        self.num_movies = args.get('num_movies')
        self.beta_annealing_schedule = args.get('beta_annealing_schedule')
        self.beta_max = args.get('beta_max')
        self.M = args.get('M')
        self.R = args.get('R')

        hidden_dimension = args.get('hidden_dimension')
        latent_dimension = args.get('latent_dimension')
        dropout_p = args.get('dropout_p')
        weight_init_type = args.get('weight_init_type')
        activation = get_activation(args.get('activation_type'))

        self.encoder = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features=self.num_movies, out_features=hidden_dimension),
            activation,
            nn.Dropout(dropout_p),
        )
        self.mu = nn.Linear(in_features=hidden_dimension, out_features=latent_dimension)
        self.logvar = nn.Linear(in_features=hidden_dimension, out_features=latent_dimension)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dimension, out_features=hidden_dimension),
            activation,
            nn.Dropout(dropout_p),
            nn.Linear(in_features=hidden_dimension, out_features=self.num_movies)
        )

        # Optionally learn user/movie bias vectors.
        if self.use_user_bias:
            self.user_bias = torch.nn.Parameter(torch.ones((self.num_users, 1)))
        if self.use_movie_bias:
            self.movie_bias = torch.nn.Parameter(torch.ones((1, self.num_movies)))
        
        # Initialize weights for linear layers.
        init_weights([self.encoder[1], self.mu, self.logvar, self.decoder[0], self.decoder[-1]], weight_init_type)

    def reparameterize(self, mu, logvar):
        '''
        Reparameterize z_u = mu_phi + epsilon * sigma_phi so that the gradient with respect to phi
        can be back-propagated through sampling z_u.
        '''
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data, users):
        h = self.encoder(F.normalize(data))
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        if self.use_user_bias:
            recon += self.user_bias[users].repeat(1, self.num_movies) 
        if self.use_movie_bias:
            recon += self.movie_bias.repeat(z.shape[0], 1)
        return recon, z, mu, logvar

    def get_beta(self, epoch, num_epochs=2000):
        '''
        Get the index beta for the KL loss term. One of constant value, linear annealing, and cyclic annealing.
        '''
        if self.beta_annealing_schedule is None:
            beta = beta_max
        elif self.beta_annealing_schedule == 'linear':
            beta = beta_max * epoch / num_epochs
        elif self.beta_annealing_schedule == 'cyclic':
            beta = min((epoch % self.M) * self.beta_max / (self.M*self.R), self.beta_max)
        return beta

    def loss_function(self, original, reconstructed, mask, mu, logvar, beta):
        MMSE = ((reconstructed - original) ** 2 * mask).sum(axis=1).mean()
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return MMSE + beta * KLD, MMSE.item(), KLD.item()