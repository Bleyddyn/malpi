""" From: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
"""
import pickle

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
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
                 beta = 4.0,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.kld_weight = beta
        self.img_dim = (in_channels, input_size, input_size) # For use by the TensorboardGenerativeModelImageSampler
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
                    #nn.BatchNorm2d(h_dim),
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
                    #nn.BatchNorm2d(hidden_dims[i + 1]),
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
                            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        if isinstance(input, tuple) or isinstance(input, list):
            input = input[0]
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
        self.z = self.reparameterize(mu, log_var)
        return  [self.decode(self.z), input, mu, log_var]

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

        kld_weight = self.kld_weight
        recons_loss =F.mse_loss(recons, input)
        #recons_loss =F.binary_cross_entropy(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + (kld_weight * kld_loss)
        return loss

    def loss_function_exp(self, target, recons, mu, log_var ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """

        kld_weight = self.kld_weight
        recons_loss =F.mse_loss(recons, target)
        #recons_loss =F.binary_cross_entropy(recons, target)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + (kld_weight * kld_loss)
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

class VAEWithAuxOuts(VanillaVAE):
    """ A VAE with a number of auxiliary outputs.
        Auxiliary outputs are:
            CTE (crosstrack error): 1D float
            Steering angle: 1D float
            Throttle: 1D float
            Track name: categorical, 9 categories
        @param aux_dims: a list of the linear layer size for each of four auxiliary outputs
    """

    def __init__(self,
                 input_size: int,
                 latent_dim: int,
                 in_channels: int = 3,
                 hidden_dims: List = None,
                 beta = 4.0,
                 aux_dims: List = [250,250,250,250],
                 **kwargs) -> None:
        super(VAEWithAuxOuts, self).__init__(input_size, latent_dim, in_channels, hidden_dims, beta)

        self.aux_dims = aux_dims
        self.build_aux()

    def build_aux(self):
        # For each aux output, we need a linear layer and an activation function
        # Steering: linear, Tanh
        # Throttle: linear, Tanh
        # CTE: linear, no activation
        # Track number: linear, softmax

        self.steering = nn.Sequential(
            nn.Linear(self.latent_dim, self.aux_dims[1]),
            nn.ReLU(),
            nn.Linear(self.aux_dims[1], 1),
            nn.Tanh()
        )
        self.throttle = nn.Sequential(
            nn.Linear(self.latent_dim, self.aux_dims[2]),
            nn.ReLU(),
            nn.Linear(self.aux_dims[2], 1),
            nn.Tanh()
        )
        self.cte = nn.Sequential(
            nn.Linear(self.latent_dim, self.aux_dims[0]),
            nn.ReLU(),
            nn.Linear(self.aux_dims[0], 1)
        )
        self.track = nn.Sequential(
            nn.Linear(self.latent_dim, self.aux_dims[3]),
            nn.ReLU(),
            nn.Linear(self.aux_dims[3], 9),
            nn.Softmax(dim=0)
        )

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        recons, _, mu, log_var = super(VAEWithAuxOuts, self).forward(input, **kwargs)
        cte_out = self.cte(self.z)
        steering_out = self.steering(self.z)
        throttle_out = self.throttle(self.z)
        track_out = self.track(self.z)
        return  [recons, mu, log_var, steering_out, throttle_out, cte_out, track_out]

    def loss_function(self, targets: Dict, outputs: Dict, mu, log_var) -> dict:
        """ Combine the VAE loss with the auxiliary losses. """
        loss = super(VAEWithAuxOuts, self).loss_function_exp(targets["images"], outputs["images"], mu, log_var )

        drive_out    = torch.cat((outputs["steering"], outputs["throttle"], outputs["cte"]), dim=1)
        drive_target = torch.cat((targets["steering"], targets["throttle"], targets["cte"]), dim=1)
        drive_loss = F.mse_loss(drive_out, drive_target)

        # Track loss
        track_out = outputs["track"]
        track_target = targets["track"]
        track_target = torch.nn.functional.one_hot(track_target, num_classes=9).float()
        track_loss = F.cross_entropy(track_out, track_target)

        return (loss, drive_loss, track_loss)

class SplitDriver(nn.Module):
    """ A DonkeyCar driver that takes as inputs mu/log_var from a
        pre-trained VAE, samples a z-space, then drives based on that. """

    def __init__(self, latent_dim, outputs=2, no_var=False):
        super(SplitDriver, self).__init__()

        self.latent_dim = latent_dim
        self.no_var = no_var
        self.meta = {}
        self.is_rnn = False

        self.driver = nn.Sequential(
            torch.nn.Linear(self.latent_dim, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, outputs),
            torch.nn.Tanh() # -1 to +1
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
        if self.no_var:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)
        return self.driver.forward(z)

class RNNDriver(nn.Module):
    """ A DonkeyCar driver that takes as inputs mu/log_var from a
        pre-trained VAE, samples a z-space, then drives based on that. """

    def __init__(self, latent_dim, batch_size, hidden_size=100, outputs=2, no_var=False):
        super(RNNDriver, self).__init__()

        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.no_var = no_var
        self.meta = {}
        self.is_rnn = True
        self.first = True

        #self.input_embed = nn.Sequential(
        #    torch.nn.Linear(self.latent_dim, 250),
        #    torch.nn.ReLU(),
        #)

        self.rnn = torch.nn.LSTM(input_size=self.latent_dim, hidden_size=self.hidden_size, batch_first=True)

        self.output_embed = nn.Sequential(
            torch.nn.Linear(self.hidden_size, outputs),
            torch.nn.Tanh() # -1 to +1
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

    def init_hc(self, device):
        #h = torch.zeros(1,self.hidden_size)
        #c = torch.zeros(1,self.hidden_size)
        #rnn_state = torch.zeros(2,1,self.batch_size, self.hidden_size, device=device)
        h = torch.zeros(1,self.batch_size, self.hidden_size, device=device)
        c = torch.zeros(1,self.batch_size, self.hidden_size, device=device)
        rnn_state = (h, c)
        return rnn_state

    def forward(self, mu: Tensor, log_var: Tensor, hidden, cell, **kwargs) -> List[Tensor]:
        if self.first:
            print( f"RNNDriver.forward mu: {mu.shape}" )
            self.first = False
        # Input should be mu and log_var, both of length latent_dim
        if self.no_var:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)
        if hidden is None or cell is None:
            rnn_state = self.init_hc(z.device)
        else:
            rnn_state = (hidden, cell)
        out, (hidden, cell) = self.rnn.forward(z, rnn_state)
        outputs = self.output_embed.forward(out)
        return outputs, (hidden, cell)


class CombinedRNNDriver(nn.Module):

    def __init__(self, vae_model, driver, no_var=False):
        super().__init__()
        self.vae = vae_model
        self.driver = driver
        self.driver.no_var = no_var
        self.count = 0
        self.hidden = None
        self.cell = None

    def reset(self):
        self.hidden = None
        self.cell = None

    def forward(self, image: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.vae.encode(image)
        outputs, rnn_state = self.driver.forward(mu, log_var, self.hidden, self.cell)
        self.hidden, self.cell = rnn_state
        return outputs

    def predict(self, image: Tensor, **kwargs) -> List[Tensor]:
        return self.forward(image)

class CombinedDriver(nn.Module):

    def __init__(self, vae_model, driver, no_var=False):
        super().__init__()
        self.vae = vae_model
        self.driver = driver
        self.driver.no_var = no_var
        self.count = 0

    def reset(self):
        pass

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
