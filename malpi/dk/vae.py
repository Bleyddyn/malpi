""" From: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""

import torch
from torch import nn
from torch.nn import functional as F
# import load_learner from fastai
from fastai.vision.all import *

from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

from abc import abstractmethod

class PrintLayer(nn.Module):
    def forward(self, x):
        #print(x.size())
        return x

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):


    def __init__(self,
                 input_size: int,
                 latent_dim: int,
                 in_channels: int = 3,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.kld_weight = 0.00025 # TODO calculate based on: #al_img.shape[0]/ self.num_train_imgs
        self.meta = {}

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.final_img = int(input_size / (2**len(hidden_dims))) # Reduce input image size by num of conv layers
        dense_calc = int(self.final_img * self.final_img * hidden_dims[-1])
        # was: hidden_dims[-1]*4

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    PrintLayer(),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(dense_calc, latent_dim)
        self.fc_var = nn.Linear(dense_calc, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, dense_calc)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    PrintLayer(),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            PrintLayer(),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, stride=1, padding= 1),
                            PrintLayer(),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
# TODO the 512 in view needs to be pulled from the list of hidden laers
        result = result.view(-1, 512, self.final_img, self.final_img)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0][0]
        input = args[1]
        mu = args[0][2]
        log_var = args[0][3]

        kld_weight = self.kld_weight # kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class SplitDriver(nn.Module):
    """ A DonkeyCar driver that takes as inputs mu/log_var from a
        pre-trained VAE, samples a z-space, then drives based on that. """

    def __init__(self, latent_dim, outputs=2):
        super(SplitDriver, self).__init__()

        self.latent_dim = latent_dim
        self.driver = nn.Sequential(
            torch.nn.Linear(self.latent_dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, outputs),
            torch.nn.Tanh()
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return (eps * std) + mu

    def forward(self, mu: Tensor, log_var: Tensor, **kwargs) -> List[Tensor]:
        # Input should be mu and log_var, both of length latent_dim
        z = self.reparameterize(mu, log_var)
        return self.driver.forward(z)

class CombinedDriver(nn.Module):

    def __init__(self, vae_path, driver_path):
        self.vae = load_learner(vae_path)
        self.driver = load_learner(driver_path)
        self.count = 0

    def forward(self, image: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.vae.encode(image)
        outputs = self.driver.forward(mu, log_var)
        #self.count += 1
        #if self.count % 20 == 0:
        #    print(f"{mu.mean()} {mu.std()}")
        #    print(f"{log_var.mean()} {log_var.std()}")
        #    print(f"{outputs}")
        return outputs

    def predict(self, image: Tensor, **kwargs) -> List[Tensor]:
        return self.forward(image)
