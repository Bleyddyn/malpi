from typing import Optional, Tuple

import torch
from lightning.pytorch import Callback, LightningModule, Trainer, LightningDataModule
import torchvision

from malpi.dk.vis import show_vae_results, evaluate_vae, visualize_batch

class TensorboardGenerativeModelImageSampler(Callback):
    """Generates images and logs to tensorboard. Your model must implement the ``forward`` function for generation.

    Requirements::

        # model must have img_dim, latent_dim attributes
        model.img_dim = (1, 28, 28)
        model.latent_dim

        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)

    Example::

        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])

    Adapted from: https://github.com/Lightning-Universe/lightning-bolts/blob/2dfe45a4cf050f120d10981c45cfa2c785a1d5e6/pl_bolts/callbacks/vision/image_generation.py
    """

    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        data_module: LightningDataModule = None,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.data_module = data_module

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._generate_and_log(trainer, pl_module)

    def _generate_and_log(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # generate images
#        with torch.no_grad():
#            pl_module.eval()
#            images = pl_module.decode(z)
#            pl_module.train()

        originals, reconstructed, samples = evaluate_vae( self.data_module.val_dataloader(),
                    pl_module.model, self.num_samples, pl_module.device )

        images = torch.cat( [originals, reconstructed, samples], dim=0 )

        if len(images.size()) == 2:
            img_dim = pl_module.img_dim
            images = images.view(self.num_samples*3, *img_dim)

        grid = torchvision.utils.make_grid(
            tensor=images,
            nrow=self.num_samples,
            padding=self.padding,
            normalize=self.normalize,
            range=self.norm_range,
            scale_each=self.scale_each,
            pad_value=self.pad_value,
        )
        str_title = f"{pl_module.__class__.__name__}_images"
        trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

if __name__ == "__main__":
    callback = TensorboardGenerativeModelImageSampler(num_samples=3)
