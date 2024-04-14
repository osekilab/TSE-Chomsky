import torch
import torch.nn as nn
from chomsky_neural.modules.token_embedder.token_embedder import TokenEmbedder


@TokenEmbedder.register("embedding")
class Embedding(TokenEmbedder):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: torch.FloatTensor = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=False,
            sparse=sparse,
            _weight=_weight,
        )
        self.output_dim = embedding_dim
        self.input_dim = num_embeddings

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_input_dim(self) -> int:
        return self.input_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        return embedded
