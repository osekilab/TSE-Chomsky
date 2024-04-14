from typing import cast

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from chomsky_neural.modules.seq2seq_encoder.seq2seq_encoder import \
    Seq2SeqEncoder


class PytorchSeq2SeqWrapper(Seq2SeqEncoder):
    def __init__(self, module: torch.nn.Module) -> None:
        try:
            if not module.batch_first:
                raise ValueError("PytorchSeq2SeqWrapper only supports batch_first=True")
        except AttributeError:
            pass
        super().__init__()
        self._module = module
        try:
            self._is_bidirectional = cast(bool, self._module.bidirectional)
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return cast(int, self._module.input_size)

    def get_output_dim(self) -> int:
        return cast(int, self._module.hidden_size) * self._num_directions

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        batch_size, max_length, _ = inputs.size()
        lengths = mask.sum(dim=1).cpu()
        packed_sequence = pack_padded_sequence(
            inputs, lengths, batch_first=True, enforce_sorted=False
        )
        packed_sequence, state = self._module(packed_sequence)
        padded_sequence, lens_unpacked = pad_packed_sequence(
            packed_sequence, batch_first=True, total_length=max_length
        )  # (batch, seq, output_dim)
        return cast(torch.Tensor, padded_sequence)


@Seq2SeqEncoder.register("gru")
class GruSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "gru".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)


@Seq2SeqEncoder.register("lstm")
class LstmSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "lstm".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)


@Seq2SeqEncoder.register("rnn")
class RnnSeq2SeqEncoder(PytorchSeq2SeqWrapper):
    """
    Registered as a `Seq2SeqEncoder` with name "rnn".
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        module = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        super().__init__(module=module)
