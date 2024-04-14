import torch
import torch.nn as nn
from chomsky_neural.modules.seq2seq_encoder.seq2seq_encoder import Seq2SeqEncoder
from chomsky_neural.modules.stack_rnn import StackRNN, StackRnnState


@Seq2SeqEncoder.register("stack_rnn")
class StackRNNSeq2SeqEncoder(Seq2SeqEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        dropout: float = 0.0,
        batch_first: bool = True,
        stack_cell_size: int = 8,
        stack_size: int = 30,
        n_stacks: int = 1,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bias = bias
        self._dropout = dropout
        self._batch_first = batch_first
        self._stack_cell_size = stack_cell_size
        self._stack_size = stack_size
        self._n_stacks = n_stacks

        inner_core = nn.RNN(
            input_size=input_size + n_stacks * stack_cell_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            batch_first=batch_first,
        )
        self._core = StackRNN(
            inner_core=inner_core,
            stack_cell_size=stack_cell_size,
            stack_size=stack_size,
            n_stacks=n_stacks,
        )

    def get_input_dim(self) -> int:
        return self._input_size

    def get_output_dim(self) -> int:
        return self._hidden_size

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: Shape (batch_size, sequence_length, input_size)
            mask: Shape (batch_size, sequence_length)
        Returns:
            output: Shape (batch_size, sequence_length, hidden_size)
        """
        batch_size = inputs.size(0)
        outputs = []

        for i in range(inputs.size(1)):
            if i == 0:
                stacks = torch.zeros(
                    batch_size, self._n_stacks, self._stack_size, self._stack_cell_size
                ).to(inputs.device)
                core_state = torch.zeros(
                    self._num_layers, batch_size, self._hidden_size
                ).to(inputs.device)
                prev_state: StackRnnState = (stacks, core_state)
            # inputs: (batch_size, input_size)
            output, prev_state = self._core(inputs[:, i, :], prev_state)
            outputs.append(output)

        # Concatenate the outputs into a tensor
        outputs = torch.stack(outputs, dim=1).squeeze(
            2
        )  # (batch_size, sequence_length, hidden_size)

        return outputs
