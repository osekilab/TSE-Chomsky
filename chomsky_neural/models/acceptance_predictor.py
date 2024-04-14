from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from chomsky_neural.data.instance import Batch
from chomsky_neural.data.vocabulary import LABEL_PAD_INDEX
from chomsky_neural.modules.datamodule.datamodule import ChomskyDatamodule
from chomsky_neural.modules.heads.classifier_head import ClassifierHead
from chomsky_neural.modules.token_embedder.token_embedder import TokenEmbedder
from chomsky_neural.torch.metrics.classification import get_classification_full_metrics
from lightning_lite.utilities.seed import seed_everything
from tango.common import Lazy
from tango.integrations.torch import LRScheduler, Optimizer

NUM_CLASSES = 2


class AcceptancePredictor(pl.LightningModule):
    def __init__(
        self,
        token_embedder: TokenEmbedder,
        classifier: ClassifierHead,
        datamodule: ChomskyDatamodule,
        optimizer: Lazy[Optimizer],
        scheduler: Optional[Lazy[LRScheduler]] = None,
        random_seed: int = 42,
    ) -> None:
        seed_everything(random_seed)
        super().__init__()
        self.token_embedder = token_embedder
        self.classifier = classifier
        self.datamodule = datamodule
        self.optimizer = optimizer.construct(params=self.parameters())
        self.scheduler = (
            scheduler.construct(optimizer=self.optimizer) if scheduler else None
        )

        self._datamodule = datamodule
        self._num_classes = NUM_CLASSES
        self._pad_idx = LABEL_PAD_INDEX

        self.train_metrics = get_classification_full_metrics(
            num_classes=self._num_classes,
            ignore_index=self._pad_idx,
            prefix="train_",
        )
        self.valid_metrics = get_classification_full_metrics(
            num_classes=self._num_classes,
            ignore_index=self._pad_idx,
            prefix="valid_",
        )
        self.test_metrics = get_classification_full_metrics(
            num_classes=self._num_classes,
            ignore_index=self._pad_idx,
            prefix="test_",
        )

    def forward(
        self,
        inputs: torch.FloatTensor,
        mask: torch.BoolTensor,
        label: Optional[torch.IntTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedded = self.token_embedder(inputs)
        return self.classifier(embedded, mask, label)

    def training_step(self, batch: Batch, batch_idx: int):
        output = self.forward(batch.inputs, batch.mask, batch.label)
        loss = output["loss"]
        assert output["logits"].shape[0] == batch.label.shape[0]
        assert output["logits"].dim() == 2
        pred = torch.argmax(F.softmax(output["logits"], dim=1), dim=1)
        assert pred.dim() == 1
        assert pred.shape[0] == batch.label.shape[0]
        return {"loss": loss, "pred": pred, "label": batch.label}

    def validation_step(self, batch: Batch, batch_idx: int):
        output = self.forward(batch.inputs, batch.mask, batch.label)
        loss = output["loss"]
        assert output["logits"].shape[0] == batch.label.shape[0]
        assert output["logits"].dim() == 2
        pred = torch.argmax(F.softmax(output["logits"], dim=1), dim=1)
        assert pred.dim() == 1
        assert pred.shape[0] == batch.label.shape[0]
        return {"loss": loss, "pred": pred, "label": batch.label}

    def test_step(self, batch: Batch, batch_idx: int):
        output = self.forward(batch.inputs, batch.mask, batch.label)
        loss = output["loss"]
        assert output["logits"].shape[0] == batch.label.shape[0]
        assert output["logits"].dim() == 2
        pred = torch.argmax(F.softmax(output["logits"], dim=1), dim=1)
        assert pred.dim() == 1
        assert pred.shape[0] == batch.label.shape[0]
        return {"loss": loss, "pred": pred, "label": batch.label}

    def predict_step(self, batch: Batch, batch_idx: int):
        output = self.forward(batch.inputs, batch.mask, batch.label)
        assert output["logits"].shape[0] == batch.label.shape[0]
        assert output["logits"].dim() == 2
        pred = torch.argmax(F.softmax(output["logits"], dim=1), dim=1)
        assert pred.dim() == 1
        assert pred.shape[0] == batch.label.shape[0]
        return pred.tolist()

    def predict(self, batch: Batch):
        output = self.forward(batch.inputs, batch.mask, batch.label)
        assert output["logits"].shape[0] == batch.label.shape[0]
        assert output["logits"].dim() == 2
        pred = torch.argmax(F.softmax(output["logits"], dim=1), dim=1)
        assert pred.dim() == 1
        assert pred.shape[0] == batch.label.shape[0]
        return pred.tolist()

    def training_step_end(self, outputs):
        self.train_metrics(outputs["pred"], outputs["label"])
        self.log("train_loss", outputs["loss"], sync_dist=True)
        self.log_dict(self.train_metrics, sync_dist=True)

    def validation_step_end(self, outputs):
        self.valid_metrics(outputs["pred"], outputs["label"])
        self.log("valid_loss", outputs["loss"], sync_dist=True)
        self.log_dict(self.valid_metrics, sync_dist=True)

    def test_step_end(self, outputs):
        self.test_metrics(outputs["pred"], outputs["label"])
        self.log("test_loss", outputs["loss"], sync_dist=True)
        self.log_dict(self.test_metrics, sync_dist=True)

    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        else:
            return self.optimizer
