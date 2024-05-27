from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from chomsky_neural.data.instance import Batch, PairBatch
from chomsky_neural.modules.seq2seq_encoder.seq2seq_encoder import Seq2SeqEncoder
from chomsky_neural.modules.token_embedder.token_embedder import TokenEmbedder
from tango.common import Lazy
from tango.integrations.torch import LRScheduler, Optimizer


class Seq2SeqWithLMHead(pl.LightningModule):
    def __init__(
        self,
        token_embedder: TokenEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        vocab_size: int,
        optimizer: Lazy[Optimizer],
        scheduler: Optional[Lazy[LRScheduler]] = None,
    ) -> None:
        super().__init__()
        self.token_embedder = token_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.vocab_size = vocab_size
        self.optimizer = optimizer.construct(params=self.parameters())
        self.scheduler = (
            scheduler.construct(optimizer=self.optimizer) if scheduler else None
        )
        self.vocab_projection = torch.nn.Linear(
            self.seq2seq_encoder.get_output_dim(),
            self.vocab_size,
        )
        assert (
            self.token_embedder.get_output_dim() == self.seq2seq_encoder.get_input_dim()
        )

    def forward(
        self,
        inputs: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> torch.FloatTensor:
        embedding = self.token_embedder(inputs)
        encoding = self.seq2seq_encoder(embedding, mask=mask)
        output = self.vocab_projection(encoding)
        return output

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> torch.Tensor:
        inputs = batch.inputs[:, :-1]
        targets = batch.inputs[:, 1:]
        mask = batch.mask[:, 1:]
        output = self(
            inputs, mask=mask
        )  # (batch_size, sequence_length -1 , vocab_size)

        # calcualte loss and perplexity
        loss = F.cross_entropy(
            output.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none"
        )
        loss *= mask.reshape(-1).float()
        mean_loss = loss.sum() / mask.sum()
        perplexity = torch.exp(mean_loss)
        perplexity_base2 = torch.pow(2, mean_loss)
        self.log_dict(
            {
                "train_loss": mean_loss.detach().cpu().item(),
                "train_perplexity": perplexity.detach().cpu().item(),
                "train_perplexity_base2": perplexity_base2.detach().cpu().item(),
            },
            on_step=True,
            on_epoch=True,
        )
        return mean_loss

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        inputs = batch.inputs[:, :-1]
        targets = batch.inputs[:, 1:]
        mask = batch.mask[:, 1:]
        output = self(inputs, mask=mask)

        # calcualte loss and perplexity
        loss = F.cross_entropy(
            output.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none"
        )
        loss *= mask.reshape(-1).float()
        mean_loss = loss.sum() / mask.sum()
        perplexity = torch.exp(mean_loss)
        perplexity_base2 = torch.pow(2, mean_loss)
        self.log_dict(
            {
                "val_loss": mean_loss.detach().cpu().item(),
                "val_perplexity": perplexity.detach().cpu().item(),
                "val_perplexity_base2": perplexity_base2.detach().cpu().item(),
            },
            on_step=True,
            on_epoch=True,
        )
        return mean_loss

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        inputs = batch.inputs[:, :-1]
        targets = batch.inputs[:, 1:]
        mask = batch.mask[:, 1:]
        output = self(inputs, mask=mask)

        # calcualte loss and perplexity
        loss = F.cross_entropy(
            output.reshape(-1, self.vocab_size), targets.reshape(-1), reduction="none"
        )
        loss *= mask.reshape(-1).float()
        mean_loss = loss.sum() / mask.sum()
        perplexity = torch.exp(mean_loss)
        perplexity_base2 = torch.pow(2, mean_loss)
        self.log_dict(
            {
                "test_loss": mean_loss.detach().cpu().item(),
                "test_perplexity": perplexity.detach().cpu().item(),
                "test_perplexity_base2": perplexity_base2.detach().cpu().item(),
            },
            on_step=True,
            on_epoch=True,
        )
        return mean_loss

    def calculate_likelihood(
        self,
        inputs: torch.FloatTensor,
        mask: torch.BoolTensor,
    ) -> List[List[float]]:
        output_list = []
        inputs_ = inputs[:, :-1]  # (batch_size, sequence_length -1)
        targets = inputs[:, 1:]
        mask_ = mask[:, 1:]
        output = self.forward(
            inputs_, mask=mask_
        )  # (batch_size, sequence_length -1 , vocab_size)
        probs = F.softmax(
            output, dim=-1
        )  # (batch_size, sequence_length -1 , vocab_size)
        probs_for_targets = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(
            -1
        )  # (batch_size, sequence_length -1)

        for i in range(probs_for_targets.shape[0]):
            instance_output = probs_for_targets[i, mask_[i]].tolist()
            instance_output = [
                -1.0
            ] + instance_output  # add -1.0 for the first [EOS] token
            assert len(instance_output) == mask[i].sum().item()
            output_list.append(instance_output)
        return output_list

    def calculate_likelihood_for_pair_batch(
        self,
        pair_batch: PairBatch,
    ) -> List[Dict[str, List[float]]]:
        output_list = []
        likelihood_pos = self.calculate_likelihood(
            pair_batch.inputs_pos, pair_batch.mask_pos
        )
        likelihood_neg = self.calculate_likelihood(
            pair_batch.inputs_neg, pair_batch.mask_neg
        )
        for likelihood_pos_, likelihood_neg_ in zip(likelihood_pos, likelihood_neg):
            output_list.append(
                {
                    "good_likelihood": likelihood_pos_,
                    "bad_likelihood": likelihood_neg_,
                }
            )
        return output_list

    def configure_optimizers(
        self,
    ) -> Optimizer | Tuple[List[Optimizer], List[LRScheduler]]:
        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        else:
            return self.optimizer
