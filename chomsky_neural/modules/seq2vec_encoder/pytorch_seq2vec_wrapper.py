from typing import cast

import torch
from chomsky_neural.modules.seq2vec_encoder.seq2vec_encoder import Seq2VecEncoder
from torch.nn.utils.rnn import pack_padded_sequence


class PytorchSeq2VecWrapper(Seq2VecEncoder):
    def __init__(self, module: torch.nn.Module) -> None:
        try:
            if not module.batch_first:
                raise ValueError("PytorchSeq2VecWrapper only supports batch_first=True")
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
        _, state = self._module(packed_sequence)
        if isinstance(state, tuple):  # for LSTM
            state = state[0]

        num_layers_times_directions, num_valid, encoding_dim = state.size()
        # if num_valid < batch_size:
        #     # batch size is the second dimension here, because pytorch
        #     # returns RNN state as a tensor of shape (num_layers * num_directions,
        #     # batch_size, hidden_size)
        #     zeros = state.new_zeros(
        #         num_layers_times_directions, batch_size - num_valid, encoding_dim
        #     )
        #     state = torch.cat([state, zeros], 1)

        state = state.transpose(0, 1)
        try:
            last_state_index = 2 if self._module.bidirectional else 1
        except AttributeError:
            last_state_index = 1
        last_layer_state = state[:, -last_state_index:, :]
        return last_layer_state.contiguous().view([-1, self.get_output_dim()])


@Seq2VecEncoder.register("gru")
class GruSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "gru".
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


@Seq2VecEncoder.register("lstm")
class LstmSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "lstm".
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


@Seq2VecEncoder.register("rnn")
class RnnSeq2VecEncoder(PytorchSeq2VecWrapper):
    """
    Registered as a `Seq2VecEncoder` with name "rnn".
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
