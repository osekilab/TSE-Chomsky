import pytorch_lightning as pl
from tango.common import Registrable


class TokenEmbedder(pl.LightningModule, Registrable):
    def get_output_dim(self) -> int:
        raise NotImplementedError
