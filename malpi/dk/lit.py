import torch
from torch.nn import functional
import lightning.pytorch as pl

from malpi.dk.vae import VanillaVAE, VAEWithAuxOuts, SplitDriver, RNNDriver


class LitVAE(pl.LightningModule):
    """ A LightningModule that creates and trains a VanillaVAE model."""

    def __init__(self, lr:float=1e-3, image_size: int=128, latent_dim: int=128, beta:float =4.0, notes: str = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = VanillaVAE(input_size=image_size, latent_dim=latent_dim, beta=beta)
        #print(self.model)
        self.lr = lr
        self.latent_dim = latent_dim
        self.img_dim = self.model.img_dim

        if notes is not None:
            self.model.meta['notes'] = notes
        self.model.meta['image_size'] = (image_size,image_size)
        #self.model.meta['epochs'] = epochs
        self.model.meta['lr'] = lr
        #self.model.meta['batch_size'] = batch_size
        self.model.meta['latent_dim'] = latent_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def encode(self, image):
        return self.model.encode(image)

    def decode(self, z):
        # For use by TensorboardGenerativeModelImageSampler
        return self.model.decode(z)

    def _run_one_batch(self, batch, batch_idx):

        recons, _, mu, log_var = self.model.forward(batch)

        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]        

        try:
            loss_vae = self.model.loss_function_exp( batch, recons, mu, log_var )
        except RuntimeError as ex:
            raise

        return recons, loss_vae

    def training_step(self, batch, batch_idx):
        outputs, train_loss = self._run_one_batch(batch, batch_idx)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs, val_loss = self._run_one_batch(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs, test_loss = self._run_one_batch(batch, batch_idx)
        self.log("test_loss", test_loss, prog_bar=True)

class LitVAEWithAux(pl.LightningModule):
    def __init__(self, lr:float=1e-3, image_size: int=128, latent_dim: int=128, beta:float =4.0, notes: str = None):
        super().__init__()
        self.save_hyperparameters()
        self.model = VAEWithAuxOuts(input_size=image_size, latent_dim=latent_dim, beta=beta)
        self.lr = lr
        self.latent_dim = latent_dim
        self.img_dim = self.model.img_dim
        self.alpha_vae = 1.0
        self.alpha_drive = 0.0
        self.alpha_track = 0.0

        if notes is not None:
            self.model.meta['notes'] = notes
        self.model.meta['image_size'] = (image_size,image_size)
        #self.model.meta['epochs'] = epochs
        self.model.meta['lr'] = lr
        #self.model.meta['batch_size'] = batch_size
        self.model.meta['latent_dim'] = latent_dim
        self.model.meta['loss_alpha'] = (self.alpha_vae, self.alpha_drive, self.alpha_track)
        # From the fastai version.
        #vae.meta['input'] = input_file
        #vae.meta['transforms'] = len(batch_tfms)
        #vae.meta['aux'] = True
        #vae.meta['train'] = len(dls.train_ds)
        #vae.meta['valid'] = len(dls.valid_ds)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def decode(self, z):
    # For use by TensorboardGenerativeModelImageSampler
        return self.model.decode(z)

    def _run_one_batch(self, batch, batch_idx):
        images, angle, throttle, cte, track = batch

        # Unsqueeze angle, throttle and cte to make them 2D tensors
        angle = angle.unsqueeze(1)
        throttle = throttle.unsqueeze(1)
        cte = cte.unsqueeze(1)

        recons, mu, log_var, steering_out, throttle_out, cte_out, track_out = self.model.forward(images)

        targets = {"steering": angle, "throttle": throttle, "cte": cte, "track": track}
        targets["images"] = images
        outputs = {"steering": steering_out, "throttle": throttle_out, "cte": cte_out, "track": track_out}
        outputs["images"] = recons

        try:
            loss_vae, loss_drive, loss_track = self.model.loss_function( targets, outputs, mu, log_var )
        except RuntimeError as ex:
            raise

        train_loss = (loss_vae * self.alpha_vae) + (loss_drive * self.alpha_drive) + (loss_track * self.alpha_track)
        return outputs, train_loss

    def training_step(self, batch, batch_idx):
        outputs, train_loss = self._run_one_batch(batch, batch_idx)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs, val_loss = self._run_one_batch(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs, val_loss = self._run_one_batch(batch, batch_idx)
        self.log("test_loss", test_loss, prog_bar=True)

class DKDriverModule(pl.LightningModule):
    def __init__(self, lr:float=1e-3, latent_dim: int=128, notes: str=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = SplitDriver(latent_dim=latent_dim)
        #print(self.model)
        self.lr = lr
        self.latent_dim = latent_dim

        if notes is not None:
            self.model.meta['notes'] = notes
        self.model.meta['lr'] = lr
        self.model.meta['latent_dim'] = latent_dim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, mu, log_var):
        return self.model(mu, log_var)

    def _run_one_batch(self, batch, batch_idx):
        mu = batch[0]
        log_var = batch[1]
        steering = torch.unsqueeze(batch[2], dim=1)
        throttle = torch.unsqueeze(batch[3], dim=1)
        outputs = self.model.forward(mu, log_var)
        #outputs, (hidden, cell) = self.model.forward(mu, log_var)
        targets = torch.cat((steering, throttle), dim=1)

        drive_loss = functional.mse_loss(outputs, targets)

        return outputs, drive_loss

    def on_train_start(self):
        if self.logger is not None and 'notes' in self.model.meta:
            self.logger.experiment.add_text("Notes", self.model.meta['notes'])

    def training_step(self, batch, batch_idx):
        outputs, train_loss = self._run_one_batch(batch, batch_idx)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs, val_loss = self._run_one_batch(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs, test_loss = self._run_one_batch(batch, batch_idx)
        self.log("test_loss", test_loss, prog_bar=True)

class DKRNNDriverModule(pl.LightningModule):
    def __init__(self, lr:float=1e-3, latent_dim: int=128, hidden_size=100, notes: str=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = RNNDriver(latent_dim, hidden_size=hidden_size, outputs=2, no_var=False)
        #print(self.model)
        self.lr = lr
        self.latent_dim = latent_dim

        if notes is not None:
            self.model.meta['notes'] = notes
        self.model.meta['lr'] = lr
        self.model.meta['latent_dim'] = latent_dim
        self.model.meta['hidden_size'] = hidden_size

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, mu, log_var, hidden, cell):
        return self.model(mu, log_var, hidden, cell)

    def _run_one_batch(self, batch, batch_idx, hidden=None, cell=None):
        mu = batch[0]
        log_var = batch[1]
        steering = torch.unsqueeze(batch[2], dim=1)
        throttle = torch.unsqueeze(batch[3], dim=1)
        outputs, (hidden, cell) = self.model.forward(mu, log_var, hidden, cell)
        targets = torch.cat((steering, throttle), dim=1)

        drive_loss = functional.mse_loss(outputs, targets)

        return outputs, drive_loss

    def on_train_start(self):
        if self.logger is not None and 'notes' in self.model.meta:
            self.logger.experiment.add_text("Notes", self.model.meta['notes'])

    def training_step(self, batch, batch_idx):
        outputs, train_loss = self._run_one_batch(batch, batch_idx)
        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs, val_loss = self._run_one_batch(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs, test_loss = self._run_one_batch(batch, batch_idx)
        self.log("test_loss", test_loss, prog_bar=True)

