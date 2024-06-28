import math
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
# TODO: GATConv has been modified to dgl.nn.pytorch
from dgl.nn.pytorch import GATConv
# from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.nn.parameter import Parameter
from torch.nn import init

from src.dlutils import *
from src.constant import Args

# from src.constants import *

torch.manual_seed(1)


## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
    def __init__(self, feats):
        super(LSTM_Univariate, self).__init__()
        self.name = 'LSTM_Univariate'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 1
        self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

    def forward(self, x):
        hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64),
                   torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
        outputs = []
        for i, g in enumerate(x):
            multivariate_output = []
            for j in range(self.n_feats):
                univariate_input = g.view(-1)[j].view(1, 1, -1)
                out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
                multivariate_output.append(2 * out.view(-1))
            output = torch.cat(multivariate_output)
            outputs.append(output)
        return torch.stack(outputs)


## Simple Multi-Head Self-Attention Model
class Attention(nn.Module):
    def __init__(self, feats):
        super(Attention, self).__init__()
        self.name = 'Attention'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5  # MHA w_size = 5
        self.n = self.n_feats * self.n_window
        self.atts = [nn.Sequential(nn.Linear(self.n, feats * feats),
                                   nn.ReLU(True)) for i in range(1)]
        self.atts = nn.ModuleList(self.atts)

    def forward(self, g):
        for at in self.atts:
            ats = at(g.view(-1)).reshape(self.n_feats, self.n_feats)
            g = torch.matmul(g, ats)
        return g, ats


## LSTM_AD Model
class LSTM_AD(nn.Module):
    def __init__(self, feats):
        super(LSTM_AD, self).__init__()
        self.name = 'LSTM_AD'
        self.lr = 0.002
        self.n_feats = feats
        self.n_hidden = 64
        self.lstm = nn.LSTM(feats, self.n_hidden)
        self.lstm2 = nn.LSTM(feats, self.n_feats)
        self.fcn = nn.Sequential(nn.Linear(self.n_feats, self.n_feats), nn.Sigmoid())

    def forward(self, x):
        hidden = (
            torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
        hidden2 = (
            torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
        outputs = []
        for i, g in enumerate(x):
            out, hidden = self.lstm(g.view(1, 1, -1), hidden)
            out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
            out = self.fcn(out.view(-1))
            outputs.append(2 * out.view(-1))
        return torch.stack(outputs)


## DAGMM Model (ICLR 18)
class DAGMM(nn.Module):
    def __init__(self, feats):
        super(DAGMM, self).__init__()
        self.name = 'DAGMM'
        self.lr = 0.0001
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 8
        self.n_window = 5  # DAGMM w_size = 5
        self.n = self.n_feats * self.n_window
        self.n_gmm = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.estimate = nn.Sequential(
            nn.Linear(self.n_latent + 2, self.n_hidden), nn.Tanh(), nn.Dropout(0.5),
            nn.Linear(self.n_hidden, self.n_gmm), nn.Softmax(dim=1),
        )

    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x - x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity

    def forward(self, x):
        ## Encode Decoder
        x = x.view(1, -1)
        z_c = self.encoder(x)
        x_hat = self.decoder(z_c)
        ## Compute Reconstructoin
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        ## Estimate
        gamma = self.estimate(z)
        return z_c, x_hat.view(-1), z, gamma.view(-1)


## OmniAnomaly Model (KDD 19)
class OmniAnomaly(nn.Module):
    def __init__(self, feats):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = 32
        self.n_latent = 8
        self.lstm = nn.GRU(feats, self.n_hidden, 2)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2 * self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x, hidden=None):
        hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        out, hidden = self.lstm(x.view(1, 1, -1), hidden)
        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        ## Decoder
        x = self.decoder(x)
        return x.view(-1), mu.view(-1), logvar.view(-1), hidden


## USAD Model (KDD 20)
class USAD(nn.Module):
    def __init__(self, feats):
        super(USAD, self).__init__()
        self.name = 'USAD'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_latent = 5
        self.n_window = 5  # USAD w_size = 5
        self.n = self.n_feats * self.n_window
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = self.encoder(g.view(1, -1))
        ## Decoders (Phase 1)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ## Encode-Decode (Phase 2)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)


## MSCRED Model (AAAI 19)
class MSCRED(nn.Module):
    def __init__(self, feats):
        super(MSCRED, self).__init__()
        self.name = 'MSCRED'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.ModuleList([
            ConvLSTM(1, 32, (3, 3), 1, True, True, False),
            ConvLSTM(32, 64, (3, 3), 1, True, True, False),
            ConvLSTM(64, 128, (3, 3), 1, True, True, False),
        ]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, (3, 3), 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        for cell in self.encoder:
            _, z = cell(z.view(1, *z.shape))
            z = z[0][0]
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
    def __init__(self, feats):
        super(CAE_M, self).__init__()
        self.name = 'CAE_M'
        self.lr = 0.001
        self.n_feats = feats
        self.n_window = feats
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Encode
        z = g.view(1, 1, self.n_feats, self.n_window)
        z = self.encoder(z)
        ## Decode
        x = self.decoder(z)
        return x.view(-1)


## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
    def __init__(self, feats):
        super(MTAD_GAT, self).__init__()
        self.name = 'MTAD_GAT'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = feats
        self.n_hidden = feats * feats
        self.g = dgl.graph((torch.tensor(list(range(1, feats + 1))), torch.tensor([0] * feats)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(feats, 1, feats)
        self.time_gat = GATConv(feats, 1, feats)
        self.gru = nn.GRU((feats + 1) * feats * 3, feats * feats, 1)

    def forward(self, data, hidden):
        hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.cat((torch.zeros(1, self.n_feats), data))
        feat_r = self.feature_gat(self.g, data_r)
        data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
        time_r = self.time_gat(self.g, data_t)
        data = torch.cat((torch.zeros(1, self.n_feats), data))
        data = data.view(self.n_window + 1, self.n_feats, 1)
        x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
        x, h = self.gru(x, hidden)
        return x.view(-1), h


## GDN Model (AAAI 21)
class GDN(nn.Module):
    def __init__(self, feats):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats
        src_ids = np.repeat(np.array(list(range(feats))), feats)
        dst_ids = np.array(list(range(feats)) * feats)
        self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
        self.g = dgl.add_self_loop(self.g)
        self.feature_gat = GATConv(1, 1, feats)
        self.attention = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
        )
        self.fcn = nn.Sequential(
            nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
        )

    def forward(self, data):
        # Bahdanau style attention
        att_score = self.attention(data).view(self.n_window, 1)
        data = data.view(self.n_window, self.n_feats)
        data_r = torch.matmul(data.permute(1, 0), att_score)
        # GAT convolution on complete graph
        feat_r = self.feature_gat(self.g, data_r)
        feat_r = feat_r.view(self.n_feats, self.n_feats)
        # Pass through a FCN
        x = self.fcn(feat_r)
        return x.view(-1)


# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
    def __init__(self, feats):
        super(MAD_GAN, self).__init__()
        self.name = 'MAD_GAN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_hidden = 16
        self.n_window = 5  # MAD_GAN w_size = 5
        self.n = self.n_feats * self.n_window
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        )

    def forward(self, g):
        ## Generate
        z = self.generator(g.view(1, -1))
        ## Discriminator
        real_score = self.discriminator(g.view(1, -1))
        fake_score = self.discriminator(z.view(1, -1))
        return z.view(-1), real_score.view(-1), fake_score.view(-1)


# Proposed Model (VLDB 22)
class TranAD_Basic(nn.Module):
    def __init__(self, feats):
        super(TranAD_Basic, self).__init__()
        self.name = 'TranAD_Basic'
        # self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sigmoid()

    def forward(self, src, tgt):
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x


# Proposed Model (FCN) + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD_Transformer(nn.Module):
    def __init__(self, feats):
        super(TranAD_Transformer, self).__init__()
        self.name = 'TranAD_Transformer'
        # self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_hidden = 8
        self.n_window = 10
        self.n = 2 * self.n_feats * self.n_window
        self.transformer_encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.ReLU(True))
        self.transformer_decoder1 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
        self.transformer_decoder2 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * feats), nn.ReLU(True))
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2).flatten(start_dim=1)
        tgt = self.transformer_encoder(src)
        return tgt

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.transformer_decoder1(self.encode(src, c, tgt))
        x1 = x1.reshape(-1, 1, 2 * self.n_feats).permute(1, 0, 2)
        x1 = self.fcn(x1)
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.transformer_decoder2(self.encode(src, c, tgt))
        x2 = x2.reshape(-1, 1, 2 * self.n_feats).permute(1, 0, 2)
        x2 = self.fcn(x2)
        return x1, x2


# Proposed Model + Self Conditioning + MAML (VLDB 22)
class TranAD_Adversarial(nn.Module):
    def __init__(self, feats):
        super(TranAD_Adversarial, self).__init__()
        self.name = 'TranAD_Adversarial'
        # self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode_decode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x = self.encode_decode(src, c, tgt)
        # Phase 2 - With anomaly scores
        c = (x - src) ** 2
        x = self.encode_decode(src, c, tgt)
        return x


# Proposed Model + Adversarial + MAML (VLDB 22)
class TranAD_SelfConditioning(nn.Module):
    def __init__(self, feats):
        super(TranAD_SelfConditioning, self).__init__()
        self.name = 'TranAD_SelfConditioning'
        # self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


# Proposed Model + Self Conditioning + Adversarial + MAML (VLDB 22)
class TranAD(nn.Module):
    def __init__(self, feats, args: Args):
        super(TranAD, self).__init__()
        self.name = args.model
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window

        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # In order to concatenate Positional encoding
        src = src * math.sqrt(self.n_feats)  # Power 1/2
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)  # expand twofold on 3rd dimension
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranVTV(nn.Module):
    """
    TranVTV model: Transformer on Variable and Time dimension via Vanilla mode (TranVTV)
    """

    def __init__(self, feats, args: Args):
        super(TranVTV, self).__init__()
        self.name = args.model
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window

        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # In order to concatenate Positional encoding
        src = src * math.sqrt(self.n_feats)  # Power 1/2
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)  # expand twofold on 3rd dimension
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranVTP(nn.Module):
    """
    TranVTP model: Transformer on Variable and Time dimension via Parallel mode (TranVTP)
    """

    def __init__(self, feats, args: Args):
        super(TranVTP, self).__init__()
        self.name = args.model
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window

        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayerParallel(d_model_v=feats, d_model_t=args.win_size, nhead=feats,
                                                         dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        decoder_layers1 = TransformerDecoderLayerParallel(d_model_v=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)

        decoder_layers2 = TransformerDecoderLayerParallel(d_model_v=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)

        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # In order to concatenate Positional encoding

        src = src * math.sqrt(self.n_feats)  # Power 1/2

        src = self.pos_encoder(src)

        memory = self.transformer_encoder(src)
        # tgt = tgt.repeat(1, 1, 2)  # expand twofold on 3rd dimension # TODO: why don't encode position?
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranVTS(nn.Module):
    """
    TranVTP model: Transformer on Variable and Time dimension via Series mode (TranVTS)
    """

    def __init__(self, feats, args: Args):
        super(TranVTS, self).__init__()
        self.name = args.model
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window

        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayerSeries(d_model_v=feats, d_model_t=args.win_size, nhead=feats,
                                                       dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        decoder_layers1 = TransformerDecoderLayerSeries(d_model_v=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)

        decoder_layers2 = TransformerDecoderLayerSeries(d_model_v=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)

        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # In order to concatenate Positional encoding

        src = src * math.sqrt(self.n_feats)  # Power 1/2

        src = self.pos_encoder(src)

        memory = self.transformer_encoder(src)
        # tgt = tgt.repeat(1, 1, 2)  # expand twofold on 3rd dimension # TODO: why don't encode position?
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class GTranVTV(nn.Module):
    """
    The model of Graph Transformer on Variable and Temporal dimension via Vanilla mode (GTranVTV)
    """

    def __init__(self, feats, args: Args):
        super(GTranVTV, self).__init__()
        # ==============================================================================================================
        self.name = args.model
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window
        # ===================================================================================================Graph Layer
        from .GDN import GDN
        from .graph_utils import get_fc_graph_struc, get_feature_map, build_loc_net
        feature_map = get_feature_map(dataset_path=os.path.join("dataset", args.dataset, "train.csv"))
        fc_struc = get_fc_graph_struc(dataset_path=os.path.join("dataset", args.dataset, "train.csv"))
        fc_edge_index = build_loc_net(struc=fc_struc, all_features=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        edge_index_sets = [fc_edge_index]
        # So strange, which is residual!!! But this list will be unpacked in next step, so we best keep the original format.
        self.gnn = GDN(
            edge_index_sets=edge_index_sets, node_num=len(feature_map),
            input_dim=args.win_size,
            dim=args.g_dim,
            mlp_out_features=args.win_size,
            out_layer_num=args.g_out_layer_num,
            out_layer_inter_dim=args.g_out_layer_inter_dim,
            topk=args.g_top_k
        )
        # =============================================================================================Transformer Layer
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # In order to concatenate Positional encoding
        src = src * math.sqrt(self.n_feats)  # Power 1/2
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)  # expand twofold on 3rd dimension
        return tgt, memory

    def forward(self, src, tgt):  # 20 32 51
        # ==============================================================================================================
        # 1. GCN
        src = src.permute(1, 2, 0)  # 32 51 20
        src = self.gnn(src)
        src = src.permute(2, 0, 1)
        # ==============================================================================================================
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class GTranVTS(nn.Module):
    """
    The model of Graph Transformer on Variable and Temporal dimension via Series mode (GTranVTS)
    """

    def __init__(self, feats, args: Args):
        super(GTranVTS, self).__init__()
        self.name = args.model
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window
        # ===================================================================================================Graph Layer
        from .GDN import GDN
        from .graph_utils import get_fc_graph_struc, get_feature_map, build_loc_net
        feature_map = get_feature_map(dataset_path=os.path.join("dataset", args.dataset, "train.csv"))
        fc_struc = get_fc_graph_struc(dataset_path=os.path.join("dataset", args.dataset, "train.csv"))
        fc_edge_index = build_loc_net(struc=fc_struc, all_features=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        edge_index_sets = [fc_edge_index]
        # So strange, which is residual!!! But this list will be unpacked in next step, so we best keep the original format.
        self.gnn = GDN(
            edge_index_sets=edge_index_sets, node_num=len(feature_map),
            input_dim=args.win_size,
            dim=args.g_dim,
            mlp_out_features=args.win_size,
            out_layer_num=args.g_out_layer_num,
            out_layer_inter_dim=args.g_out_layer_inter_dim,
            topk=args.g_top_k
        )
        # =============================================================================================Transformer Layer
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayerSeries(d_model_v=feats, d_model_t=args.win_size, nhead=feats,
                                                       dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        decoder_layers1 = TransformerDecoderLayerSeries(d_model_v=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)

        decoder_layers2 = TransformerDecoderLayerSeries(d_model_v=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)

        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # In order to concatenate Positional encoding
        src = src * math.sqrt(self.n_feats)  # Power 1/2
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        # tgt = tgt.repeat(1, 1, 2)  # expand twofold on 3rd dimension # TODO: why don't encode position?
        return tgt, memory

    def forward(self, src, tgt):
        # ==============================================================================================================
        # 1. GCN
        src = src.permute(1, 2, 0)  # 32 51 20
        src = self.gnn(src)
        src = src.permute(2, 0, 1)
        # ==============================================================================================================
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class GTranVTP(nn.Module):
    """
    The model of Graph Transformer on Variable and Temporal dimension via Parallel mode (GTranVTP)
    """

    def __init__(self, feats, args: Args):
        super(GTranVTP, self).__init__()
        self.name = args.model
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window
        # ===================================================================================================Graph Layer
        from .GDN import GDN
        from .graph_utils import get_fc_graph_struc, get_feature_map, build_loc_net
        feature_map = get_feature_map(dataset_path=os.path.join("dataset", args.dataset, "train.csv"))
        fc_struc = get_fc_graph_struc(dataset_path=os.path.join("dataset", args.dataset, "train.csv"))
        fc_edge_index = build_loc_net(struc=fc_struc, all_features=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        edge_index_sets = [fc_edge_index]
        # So strange, which is residual!!! But this list will be unpacked in next step, so we best keep the original format.
        self.gnn = GDN(
            edge_index_sets=edge_index_sets, node_num=len(feature_map),
            input_dim=args.win_size,
            dim=args.g_dim,
            mlp_out_features=args.win_size,
            out_layer_num=args.g_out_layer_num,
            out_layer_inter_dim=args.g_out_layer_inter_dim,
            topk=args.g_top_k
        )
        # =============================================================================================Transformer Layer
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)

        encoder_layers = TransformerEncoderLayerParallel(d_model_v=feats, d_model_t=args.win_size, nhead=feats,
                                                         dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        decoder_layers1 = TransformerDecoderLayerParallel(d_model_v=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)

        decoder_layers2 = TransformerDecoderLayerParallel(d_model_v=feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)

        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # In order to concatenate Positional encoding

        src = src * math.sqrt(self.n_feats)  # Power 1/2

        src = self.pos_encoder(src)

        memory = self.transformer_encoder(src)
        # tgt = tgt.repeat(1, 1, 2)  # expand twofold on 3rd dimension # TODO: why don't encode position?
        return tgt, memory

    def forward(self, src, tgt):
        # ==============================================================================================================
        # 1. GCN
        src = src.permute(1, 2, 0)  # 32 51 20
        src = self.gnn(src)
        src = src.permute(2, 0, 1)
        # ==============================================================================================================
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class GumbelGeneratorOld(nn.Module):
    def __init__(self, sz=10, temp=10, temp_drop_frac=0.9999, use_cuda=False):
        super(GumbelGeneratorOld, self).__init__()
        self.sz = sz
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
        self.edge_weight_matrix = Parameter(torch.rand(sz, sz))
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac
        self.use_cuda = use_cuda

    def drop_temp(self):
        # drop temperature
        self.temperature = self.temperature * self.temp_drop_frac

    # output: a matrix
    def sample_all(self, hard=False, epoch=1):
        self.logp = self.gen_matrix.view(-1, 2)
        out = self.gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
        if self.use_cuda:
            out = out.cuda()
        out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        # 1000 50
        # if epoch > 998:
        #     for i in range(out_matrix.size()[0]):
        #         for j in range(out_matrix.size()[1]):
        #             if out_matrix[i][j].item() == 1:
        #                 out_matrix[j][i] = 1
        return out_matrix

    # output: the i-th column of matrix
    def sample_adj_i(self, i, hard=False, sample_time=1):
        self.logp = self.gen_matrix[:, i]
        out = self.gumbel_softmax(self.logp, self.temperature, hard=hard)
        if self.use_cuda:
            out = out.cuda()
        if hard:
            out_matrix = out.float()
        else:
            out_matrix = out[:, 0]
        return out_matrix

    def get_temperature(self):
        return self.temperature

    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)

    # ==================================================================================================================
    def gumbel_sample(self, shape, eps=1e-20):
        u = torch.rand(shape)
        gumbel = - np.log(- np.log(u + eps) + eps)
        if self.use_cuda:
            gumbel = gumbel.cuda()
        return gumbel

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        y = logits + self.gumbel_sample(logits.size())
        return torch.nn.functional.softmax(y / temperature, dim=1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            k = logits.size()[-1]
            y_hard = torch.max(y.data, 1)[1]
            y = y_hard
        return y

    @staticmethod
    def get_shortest_path_dis(adj: torch.Tensor) -> torch.Tensor:
        """
        Compute the shortest path distance between every pair of nodes in the graph.
        :param adj: 2D torch.Tensor, adjacency
        :return: 2D torch.Tensor, shortest path distance matrix
        """
        num_vertices = adj.shape[0]

        # 初始化距离矩阵
        dist_matrix = adj.clone().float()

        # 用无穷大初始化无边的地方
        inf = float('inf')
        dist_matrix[dist_matrix == 0] = inf
        dist_matrix[torch.eye(num_vertices).bool()] = 0

        # Floyd-Warshall算法
        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]

        # 将无穷大转换回-1表示无路径
        dist_matrix[dist_matrix == inf] = -1

        return dist_matrix

    @staticmethod
    def initialize_distance_matrix(adj: torch.Tensor) -> torch.Tensor:
        """
        Initialize the distance matrix. If there is an edge between i and j,
        the distance is the edge weight. Otherwise, the distance is infinity.
        """
        inf = float('inf')
        distance_matrix = adj.clone().float()  # Make a copy of the adjacency matrix and ensure it's a float tensor
        distance_matrix[adj == 0] = inf  # Set the distance to infinity where there is no edge
        distance_matrix[torch.arange(adj.size(0)), torch.arange(adj.size(0))] = 0  # Distance to self is 0
        return distance_matrix

    @staticmethod
    def floyd_warshall_worker(distance_matrix, k):
        """
        A worker function that updates the distance matrix for a given intermediate node k.
        """
        num_nodes = distance_matrix.shape[0]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if distance_matrix[i, j] > distance_matrix[i, k] + distance_matrix[k, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]
        return distance_matrix

    @staticmethod
    def floyd_warshall_parallel(adj: torch.Tensor) -> torch.Tensor:
        """
        Compute the shortest path distances using the Floyd-Warshall algorithm in parallel.
        """
        distance_matrix = GumbelGeneratorOld.initialize_distance_matrix(adj)
        num_nodes = distance_matrix.shape[0]

        with ThreadPoolExecutor() as executor:
            for k in range(num_nodes):
                future = executor.submit(GumbelGeneratorOld.floyd_warshall_worker, distance_matrix, k)
                distance_matrix = future.result()

        distance_matrix[distance_matrix == float('inf')] = -1
        return distance_matrix


class GumbelGraphormer(nn.Module):
    def __init__(self, feats, args: Args):
        super(GumbelGraphormer, self).__init__()
        self.name = 'GumbelGraphormer'
        self.args = args
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window
        self.global_step = 0
        self.spd = None

        self.graph_learning = GumbelGeneratorOld(sz=self.n_feats, temp=args.temp, temp_drop_frac=args.temp_drop_frac,
                                                 use_cuda=True)
        self.src_norm = nn.LayerNorm(self.n_feats)
        self.tgt_norm = nn.LayerNorm(self.n_feats)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayerGraph(d_model=args.win_size, nhead=args.win_size, dim_feedforward=16,
                                                      dropout=0.1)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        self.transformer_encoder = encoder_layers
        decoder_layers = TransformerDecoderLayerGraph(d_model=args.win_size, nhead=args.win_size, dim_feedforward=16,
                                                      dropout=0.1)
        # self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        self.transformer_decoder = decoder_layers
        self.fcn = nn.Sigmoid()

    def forward(self, src, tgt):
        adj = self.graph_learning.sample_all(hard=True)  # get an adj
        out_degree = torch.sum(adj, dim=1)
        in_degree = torch.sum(adj, dim=0)

        centrality = out_degree + in_degree

        # Min-Max 归一化
        centrality_min = centrality.min()
        centrality_max = centrality.max()
        centrality_norm = (centrality - centrality_min) / (centrality_max - centrality_min)

        # 将归一化后的中心性应用到 src 和 tgt
        centrality_src = src + centrality_norm
        centrality_tgt = tgt + centrality_norm

        src = centrality_src
        tgt = centrality_tgt

        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)

        if self.args.is_test and self.spd is None:
            self.spd = self.graph_learning.floyd_warshall_parallel(adj=adj)

        if (self.global_step == 0 or self.global_step % 1000 == 0) and not self.args.is_test:
            self.spd = self.graph_learning.floyd_warshall_parallel(adj=adj)
        edge_weight = self.graph_learning.edge_weight_matrix
        memory = self.transformer_encoder(src, adj, self.spd, edge_weight)
        x = self.transformer_decoder(tgt, memory, adj, self.spd, edge_weight)
        x = self.fcn(x)
        return x


class GumbelTranVTV(nn.Module):
    """
    GumbelTranVTV model: Transformer on Variable and Time dimension via Vanilla mode (TranVTV)
    """

    def __init__(self, feats, args: Args):
        super(GumbelTranVTV, self).__init__()
        self.name = args.model
        self.lr = args.lr
        self.batch = args.batch_size
        self.n_feats = feats
        self.n_window = args.win_size
        self.n = self.n_feats * self.n_window
        self.global_step = 0
        self.spd = None

        self.graph_learning = GumbelGeneratorOld(sz=self.n_feats, temp=args.temp, temp_drop_frac=args.temp_drop_frac,
                                                 use_cuda=True)
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=3 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        # self.transformer_encoder = encoder_layers
        decoder_layers1 = TransformerDecoderLayer(d_model=3 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        # self.transformer_decoder1 = decoder_layers1
        decoder_layers2 = TransformerDecoderLayer(d_model=3 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        # self.transformer_decoder2 = decoder_layers2
        self.fcn = nn.Sequential(nn.Linear(3 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)  # In order to concatenate Positional encoding
        src = src * math.sqrt(self.n_feats)  # Power 1/2
        src = self.pos_encoder(src)

        adj = self.graph_learning.sample_all(hard=True)  # get an adj
        out_degree = torch.sum(adj, dim=1)
        in_degree = torch.sum(adj, dim=0)

        centrality = out_degree + in_degree

        # Min-Max 归一化
        centrality_min = centrality.min()
        centrality_max = centrality.max()
        centrality_norm = (centrality - centrality_min) / (centrality_max - centrality_min)

        centrality_norm_expanded = centrality_norm.unsqueeze(0).unsqueeze(0)

        # 广播 `centrality_norm_expanded`
        centrality_norm_src = centrality_norm_expanded.expand(src.shape[0], src.shape[1], centrality_norm.shape[0])
        centrality_norm_tgt = centrality_norm_expanded.expand(tgt.shape[0], tgt.shape[1], centrality_norm.shape[0])

        # 拼接 `src` 和 `centrality_norm_broadcasted`
        centrality_src = torch.cat((src, centrality_norm_src), dim=2)
        tgt = tgt.repeat(1, 1, 2)  # expand twofold on 3rd dimension
        centrality_tgt = torch.cat((tgt, centrality_norm_tgt), dim=2)
        # 将归一化后的中心性应用到 src 和 tgt

        memory = self.transformer_encoder(centrality_src)

        return centrality_tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2
