from typing import Iterator, List, Optional, cast

import pytorch_lightning as pl
import torch
from chomsky_neural.data.instance import Batch, Instance, Sentence
from chomsky_neural.data.source import DataSource
from chomsky_neural.data.vocabulary import (
    BOS_TOKEN,
    EOS_TOKEN,
    LABEL_PAD_INDEX,
    PAD_TOKEN,
    UNK_TOKEN,
    Vocabulary,
)


class ChomskyDataset(torch.utils.data.Dataset[Instance]):
    def __init__(self) -> None:
        self._dataset: List[Instance] = []

    def add(self, instance: Instance) -> None:
        self._dataset.append(instance)

    def __getitem__(self, idx: int) -> Instance:
        return self._dataset[idx]

    def __len__(self) -> int:
        return len(self._dataset)


class ChomskyDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_datasource,
        val_datasource=None,
        test_datasource=None,
        batch_size=128,
        num_workers=0,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        super().__init__()
        self._train_dataset = ChomskyDataset()
        self._val_dataset = ChomskyDataset() if val_datasource else None
        self._test_dataset = ChomskyDataset() if test_datasource else None
        self._train_datasource = train_datasource
        self._val_datasource = val_datasource
        self._test_datasource = test_datasource
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab = vocab or Vocabulary()
        self.prepare_data_per_node = True

    def prepare_data(self) -> None:
        ...

    def _setup_dataset(self, dataset: ChomskyDataset, data_source: DataSource) -> None:
        for sentence in data_source.collect():
            instance = self._prepare_instance(sentence)
            dataset.add(instance)

    def build_tokens_vocab(
        self, sentences: Optional[Iterator[Sentence]] = None
    ) -> None:
        self.vocab.add(UNK_TOKEN, "tokens")
        self.vocab.add(PAD_TOKEN, "tokens")
        self.vocab.add(BOS_TOKEN, "tokens")
        self.vocab.add(EOS_TOKEN, "tokens")
        if sentences is None:
            sentences = self._train_datasource.collect()
        for sentence in sentences:
            for token in sentence.sequences:
                self.vocab.add(token, "tokens")

    def build_vocab(self, sentences: Optional[Iterator[Sentence]] = None) -> None:
        if not self.vocab.size("tokens"):
            self.build_tokens_vocab(sentences)

    def _prepare_instance(self, sentence: Sentence) -> Instance:
        inputs = (
            [self.vocab.get(BOS_TOKEN, "tokens")]
            + [self.vocab.get(token, "tokens") for token in sentence.sequences]
            + [self.vocab.get(EOS_TOKEN, "tokens")]
        )
        label = int(sentence.label)
        return Instance(inputs=inputs, label=label)

    def setup(self, stage: str) -> None:
        if not self.vocab.size("tokens"):
            self.build_vocab()
        if stage == "fit" or stage is None:
            self._setup_dataset(self._train_dataset, self._train_datasource)
            if self._val_datasource is not None:
                self._setup_dataset(self._val_dataset, self._val_datasource)
        if stage == "test":
            assert self._test_datasource is not None
            self._setup_dataset(self._test_dataset, self._test_datasource)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._val_dataset)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._test_dataset)

    def build_dataloader(
        self,
        dataset: ChomskyDataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            collate_fn=self.batch_collator,
            num_workers=self.num_workers,
        )

    def build_dataloader_from_source(
        self,
        datasource: DataSource,
        batch_size=None,
        shuffle=False,
        num_workers=0,
    ) -> torch.utils.data.DataLoader:
        dataset = ChomskyDataset()
        self._setup_dataset(dataset, datasource)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            collate_fn=self.batch_collator,
            num_workers=self.num_workers,
        )

    def batch_collator(self, instances: List[Instance]) -> Batch:
        max_length = max(len(instance.inputs) for instance in instances)
        inputs = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length),
                self.vocab.get(PAD_TOKEN, "tokens"),
                dtype=torch.long,
            ),
        )
        label = (
            cast(
                torch.IntTensor,
                torch.full((len(instances),), LABEL_PAD_INDEX, dtype=torch.int32),
            )
            if instances[0].label is not None
            else None
        )
        mask = cast(
            torch.BoolTensor,
            torch.zeros((len(instances), max_length), dtype=torch.bool),
        )

        for i, instance in enumerate(instances):
            inputs[i, : len(instance.inputs)] = torch.tensor(
                instance.inputs, dtype=torch.long
            )
            mask[i, : len(instance.inputs)] = True
            if label is not None:
                label[i] = instance.label

        assert inputs.shape == mask.shape

        return Batch(inputs=inputs, mask=mask, label=label)
