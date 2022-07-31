import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from itertools import zip_longest

class MyEmbeddingNet(nn.Module):
    def __init__(self, n_users, n_movies, n_factors_u=50, n_factors_m=50, hidden_trans=10, dropouts_trans=0.2, feature_dropout=0.05):
        super().__init__()
        self.hidden_trans = get_list(hidden_trans)
        self.dropouts_trans = get_list(dropouts_trans)
        n_last = self.hidden_trans[-1]
        self.u = nn.Embedding(n_users, n_factors_u)
        self.m = nn.Embedding(n_movies, n_factors_m)
        self.trans_u_1 = nn.Sequential(*list(self.gen_trans_layers(n_factors_u)))
        self.trans_m_1 = nn.Sequential(*list(self.gen_trans_layers(n_factors_m)))
        self.trans_u_2 = nn.Sequential(*list(self.gen_trans_layers(n_factors_u)))
        self.trans_m_2 = nn.Sequential(*list(self.gen_trans_layers(n_factors_m)))
        self.feature_dropout = nn.Dropout(feature_dropout)

    def gen_trans_layers(self, n_in):
        assert len(self.dropouts_trans) <= len(self.hidden_trans) + 1
        if len(self.dropouts_trans) > len(self.hidden_trans):
            yield nn.Dropout(self.dropouts_trans[0])
            for n_out, rate in zip_longest(self.hidden_trans, self.dropouts_trans[1:]):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out 
        else:
            for n_out, rate in zip_longest(self.hidden_trans, self.dropouts_trans):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out

    def forward(self, users, movies, minmax=None):
        users_embedding = self.u(users)
        movies_embedding = self.m(movies)
        users_trans = self.trans_u_1(users_embedding) + users_embedding
        movies_trans = self.trans_m_1(movies_embedding) + movies_embedding
        users_trans_mul = self.trans_u_2(users_embedding)
        movies_trans_mul = self.trans_m_2(movies_embedding)
        multiplication = torch.mul(users_trans_mul, movies_trans_mul)
        concatenation = torch.cat([users_trans, movies_trans], dim=1)
        ans = torch.cat([multiplication, concatenation], dim=1)
        ans = self.feature_dropout(ans)
        return ans

class MyJudgeNet(nn.Module):
    def __init__(self, embedding_length, hidden=10, dropouts=0.2):
        super().__init__()
        self.hidden = get_list(hidden)
        self.dropouts = get_list(dropouts)
        n_last = hidden[-1]
        self.judge = nn.Sequential(*list(self.gen_layers(embedding_length)))
        self.fc = nn.Linear(n_last, 1)

    def gen_layers(self, n_in):
        assert len(self.dropouts) <= len(self.hidden)
        for n_out, rate in zip_longest(self.hidden, self.dropouts):
            yield nn.Linear(n_in, n_out)
            yield nn.ReLU()
            if rate is not None and rate > 0.:
                yield nn.Dropout(rate)
            n_in = n_out

    def forward(self, embedding_vector, minmax=None):
        output = self.judge(embedding_vector)
        output = self.fc(output)
        return output

class NCF(nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.MyEmbeddingNet = MyEmbeddingNet(n_users=args.get('num_users'), n_movies=args.get('num_movies'), n_factors_u=args.get('user_code_length_ncf'),
            n_factors_m=args.get('movie_code_length_ncf'), hidden_trans=args.get('hidden_embeddingnet_ncf'), dropouts_trans=args.get('dropouts_embeddingnet_ncf'), 
            feature_dropout=args.get('feature_dropout_ncf'))
        self.judgenet = MyJudgeNet(embedding_length = args.get('user_code_length_ncf') + args.get('movie_code_length_ncf') + args.get('multiplication_code_length_ncf'), 
            hidden=args.get('hidden_judgenet_ncf'), dropouts=args.get('dropouts_judgenet_ncf'))
    def forward(self, users, movies):
        return self.judgenet(self.MyEmbeddingNet(users, movies))
