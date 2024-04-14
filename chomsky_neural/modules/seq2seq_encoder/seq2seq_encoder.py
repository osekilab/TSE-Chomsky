import pytorch_lightning as pl
from tango.common import Registrable


class Seq2SeqEncoder(pl.LightningModule, Registrable):
    def get_input_dim(self) -> int:
        raise NotImplementedError

    def get_output_dim(self) -> int:
        raise NotImplementedError

    def is_bidirectional(self) -> bool:
        raise NotImplementedError
