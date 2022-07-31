import torch
from torch import nn
import torch.nn.functional as F
from utils import init_weights, get_activation

class AE(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.use_user_bias = args.get('use_user_bias')
        self.use_movie_bias = args.get('use_movie_bias')
        self.num_users = args.get('num_users')
        self.num_movies = args.get('num_movies')

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
            nn.Linear(in_features=hidden_dimension, out_features=latent_dimension),
        )

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
        init_weights([self.encoder[1], self.encoder[-1], self.decoder[0], self.decoder[-1]], weight_init_type)

    def forward(self, data, users):
        z = self.encoder(F.normalize(data))
        recon = self.decoder(z)
        if self.use_user_bias:
            recon += self.user_bias[users].repeat(1, self.num_movies) 
        if self.use_movie_bias:
            recon += self.movie_bias.repeat(z.shape[0], 1)
        return recon, z

    def loss_function(self, original, reconstructed, mask):
        return torch.mean(mask * (original - reconstructed) ** 2)