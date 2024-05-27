from typing import Optional, Union

import torch
import torch.nn as nn
from chomsky_neural.modules.seq2seq_encoder.seq2seq_encoder import Seq2SeqEncoder
from chomsky_neural.torch.util import add_positional_features


@Seq2SeqEncoder.register("pytorch_transformer_encoder")
class PytorchTransformerEncoder(Seq2SeqEncoder):
    """
    # Parameters
    input_dim : `int`, required.
        The input dimension of the encoder.
    feedforward_hidden_dim : `int`, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : `int`, required.
        The number of stacked self attention -> feedforward -> layer normalisation blocks.
    num_attention_heads : `int`, required.
        The number of attention heads to use per layer.
    use_positional_encoding : `bool`, optional, (default = `True`)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : `float`, optional, (default = `0.1`)
        The dropout probability for the feedforward network.
    """  # noqa

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        feedforward_hidden_dim: int = 2048,
        num_attention_heads: int = 8,
        positional_encoding: Optional[str] = None,
        positional_embedding_size: int = 512,
        dropout_prob: float = 0.1,
        activation: str = "relu",
        auto_regressive: bool = False,
    ) -> None:
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout_prob,
            activation=activation,
        )
        self._transformer = nn.TransformerEncoder(layer, num_layers)
        self._input_dim = input_dim
        self._num_heads = num_attention_heads
        self._auto_regressive = auto_regressive

        # initialize parameters
        # We do this before the embeddings are initialized so we get the default initialization for the embeddings.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if positional_encoding is None:
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = None
        elif positional_encoding == "sinusoidal":
            self._sinusoidal_positional_encoding = True
            self._positional_embedding = None
        elif positional_encoding == "embedding":
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = nn.Embedding(
                positional_embedding_size, input_dim
            )
        else:
            raise ValueError(
                "positional_encoding must be one of None, 'sinusoidal', or 'embedding'"
            )

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    def is_bidirectional(self):
        return False

    def generate_square_subsequent_mask(
        self, batch_size: int, src_len: int, device: Union[torch.device, str] = "cpu"
    ) -> torch.Tensor:
        return torch.triu(
            torch.full((src_len, src_len), float("-inf"), device=device), diagonal=1
        ).repeat(batch_size * self._num_heads, 1, 1)

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor):
        output = inputs
        if self._sinusoidal_positional_encoding:
            output = add_positional_features(output)
        if self._positional_embedding is not None:
            position_ids = torch.arange(
                inputs.size(1), dtype=torch.long, device=output.device
            )
            position_ids = position_ids.unsqueeze(0).expand(inputs.shape[:-1])
            output = output + self._positional_embedding(position_ids)

        # For some reason the torch transformer expects the shape (sequence, batch, features), not the more
        # familiar (batch, sequence, features), so we have to fix it.
        output = output.permute(1, 0, 2)
        # For some other reason, the torch transformer takes the mask backwards.
        mask = ~mask
        src_mask = (
            self.generate_square_subsequent_mask(
                inputs.shape[0], inputs.shape[1], device=output.device
            )
            if self._auto_regressive
            else None
        )
        # output = self._transformer(output, mask=src_mask, src_key_padding_mask=mask)
        output = self._transformer(output, mask=src_mask)
        output = output.permute(1, 0, 2)

        return output
