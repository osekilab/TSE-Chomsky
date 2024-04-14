from typing import Dict, Iterator, List, Optional, cast

import pytorch_lightning as pl
import torch

from chomsky_neural.data.instance import PairBatch, PairInstance, Pair
from chomsky_neural.data.source import ChomskyPairDataSource
from chomsky_neural.data.vocabulary import (BOS_TOKEN, EOS_TOKEN,
                                            LABEL_PAD_INDEX, PAD_TOKEN,
                                            UNK_TOKEN, Vocabulary)


class ChomskyPairDataset(torch.utils.data.Dataset[PairInstance]):
    def __init__(self) -> None:
        self._dataset: List[PairInstance] = []

    def add(self, pair_instance: PairInstance) -> None:
        self._dataset.append(pair_instance)

    def __getitem__(self, idx: int) -> PairInstance:
        return self._dataset[idx]

    def __len__(self) -> int:
        return len(self._dataset)


class ChomskyPairDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        datasource: ChomskyPairDataSource | None = None,
        batch_size=128,
        num_workers=0,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        super().__init__()
        self._dataset = ChomskyPairDataset() if datasource is None else None
        self._datasource = datasource
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab = vocab or Vocabulary()
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        ...

    def _setup_dataset(self, dataset: ChomskyPairDataset, data_source: ChomskyPairDataSource) -> None:
        for pair in data_source.collect():
            pair_instance = self._prepare_pair(pair)
            dataset.add(pair_instance)

    def _prepare_pair(self, pair: Pair) -> PairInstance:
        inputs_pos = (
            [self.vocab.get(BOS_TOKEN, "tokens")]
            + [self.vocab.get(token, "tokens") for token in pair.sequences_pos]
            + [self.vocab.get(EOS_TOKEN, "tokens")]
        )
        inputs_neg = (
            [self.vocab.get(BOS_TOKEN, "tokens")]
            + [self.vocab.get(token, "tokens") for token in pair.sequences_neg]
            + [self.vocab.get(EOS_TOKEN, "tokens")]
        )
        return PairInstance(
            inputs_pos=inputs_pos,
            inputs_neg=inputs_neg,
        )

    def setup(self, stage: str) -> None:
        ...

    def build_dataloader_from_source(
        self,
        datasource: ChomskyPairDataSource,
        batch_size=None,
        shuffle=False,
        num_workers=0,
    ) -> torch.utils.data.DataLoader:
        dataset = ChomskyPairDataset()
        self._setup_dataset(dataset, datasource)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            collate_fn=self.batch_collator,
            num_workers=self.num_workers,
        )

    def batch_collator(self, instances: List[PairInstance]) -> PairBatch:
        max_length_pos = max(len(pair_instance.inputs_pos) for pair_instance in instances)
        max_length_neg = max(len(pair_instance.inputs_neg) for pair_instance in instances)
        inputs_pos = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length_pos),
                self.vocab.get(PAD_TOKEN, "tokens"),
                dtype=torch.long,
            ),
        )
        inputs_neg = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length_neg),
                self.vocab.get(PAD_TOKEN, "tokens"),
                dtype=torch.long,
            ),
        )
        mask_pos = cast(
            torch.BoolTensor,
            torch.zeros((len(instances), max_length_pos), dtype=torch.bool),
        )
        mask_neg = cast(
            torch.BoolTensor,
            torch.zeros((len(instances), max_length_neg), dtype=torch.bool),
        )

        for i, pair_instance in enumerate(instances):
            inputs_pos[i, : len(pair_instance.inputs_pos)] = torch.tensor(
                pair_instance.inputs_pos, dtype=torch.long
            )
            inputs_neg[i, : len(pair_instance.inputs_neg)] = torch.tensor(
                pair_instance.inputs_neg, dtype=torch.long
            )
            mask_pos[i, : len(pair_instance.inputs_pos)] = True
            mask_neg[i, : len(pair_instance.inputs_neg)] = True
        assert inputs_pos.shape == mask_pos.shape
        assert inputs_neg.shape == mask_neg.shape

        return PairBatch(
            inputs_pos=inputs_pos,
            inputs_neg=inputs_neg,
            mask_pos=mask_pos,
            mask_neg=mask_neg,
        )
