import numpy as np
import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from .vae_base import VAE


class InvertedVae(VAE):
    def __init__(self, device, img_shape, h_dim, z_dim, analytic_kl, mean_img):
        super().__init__(device, z_dim, analytic_kl)
        #  import pdb; pdb.set_trace()
        x_dim = np.prod(img_shape)
        self.img_shape = img_shape
        self.proc_data = lambda x: x.to(device).reshape(-1, x_dim)

        self.encoder_mlps = nn.ModuleList([])
        self.encoder_mus = nn.ModuleList([])
        self.encoder_sigs = nn.ModuleList([])
        self.combiner_mlps = nn.ModuleList([])
        self.previous_sample_mlps = nn.ModuleList([])
        self.z_dim = z_dim

        split_dim = h_dim // 2

        self.first_encoder = nn.Sequential(
                                nn.Linear(x_dim, h_dim), nn.Tanh(),
                                nn.Linear(h_dim, h_dim), nn.Tanh()
                            )
        self.first_mu = nn.Linear(h_dim, 1)
        self.first_sig = nn.Linear(h_dim, 1)

        for i in range(z_dim - 1):
            self.encoder_mlps.append(
                nn.Sequential(
                    nn.Linear(x_dim, split_dim), nn.Tanh(),
                    nn.Linear(split_dim, split_dim), nn.Tanh()
                ))
            self.previous_sample_mlps.append(
                nn.Sequential(
                    nn.Linear(1, split_dim), nn.Tanh(),
                    nn.Linear(split_dim, split_dim), nn.Tanh()
                ))
            self.encoder_mus.append(
                nn.Linear(h_dim, 1)
            )
            self.encoder_sigs.append(
                nn.Linear(h_dim, 1)
            )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, x_dim))  # using Bern(logit) is equivalent to putting sigmoid here.

        self.apply(self.init)
        mean_img = np.clip(mean_img, 1e-8, 1. - 1e-7)
        mean_img_logit = np.log(mean_img / (1. - mean_img))
        self.decoder[-1].bias = torch.nn.Parameter(torch.Tensor(mean_img_logit))

    def init(self, module):
        if type(module) == nn.Linear:
            torch.nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
            module.bias.data.fill_(.01)

    def encode(self, x):
        # TODO maybe fix this
        mean_n = 1
        imp_n = 1
        x = self.proc_data(x)
        mus = []
        sigs = []
        z_samples = []

        # first do the first z
        fh = self.first_encoder(x)
        fm = self.first_mu(fh)
        fs = nn.functional.softplus(self.first_sig(fh))
        current_z_sample = Normal(fm, fs).rsample(torch.Size([mean_n, imp_n]))

        z_samples.append(current_z_sample)
        mus.append(fm)
        sigs.append(fs)

        # next do all the remaining z
        all_hs = [self.encoder_mlps[i](x) for i in range(self.z_dim - 1)]
        for i in range(self.z_dim - 1):
            current_h = all_hs[i]
            current_sample_h = self.previous_sample_mlps[i](current_z_sample.squeeze(0).squeeze(0))
        
            full_h = torch.cat((current_h, current_sample_h), dim=1)
            
            curr_mu = self.encoder_mus[i](full_h)
            curr_std = nn.functional.softplus(self.encoder_sigs[i](full_h))

            current_z_sample = Normal(curr_mu, curr_std).rsample(torch.Size([mean_n, imp_n]))
            z_samples.append(current_z_sample)
            mus.append(curr_mu)
            sigs.append(curr_std)

        all_samples = torch.cat(z_samples, dim=3)
        all_mus = torch.cat(mus, dim=1)
        all_sigs = torch.cat(sigs, dim=1)
        return Normal(all_mus, all_sigs), all_samples  # torch.exp(.5 * _std)

    def decode(self, z):
        x = self.decoder(z)
        return Bernoulli(logits=x)

    def lpxz(self, true_x, x_dist):
        return x_dist.log_prob(true_x).sum(-1)

    def sample(self, num_samples=64):
        z = self.prior.sample((num_samples,))
        x_dist = self.decode(z)
        return x_dist.sample().view(num_samples, *self.img_shape)
