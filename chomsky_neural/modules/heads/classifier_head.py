from typing import Dict, Optional

import torch
from chomsky_neural.data.vocabulary import Vocabulary
from chomsky_neural.modules.feedforward import FeedForward
from chomsky_neural.modules.heads.head import Head
from chomsky_neural.modules.seq2vec_encoder.seq2vec_encoder import Seq2VecEncoder


@Head.register("classifier")
class ClassifierHead(Head):
    def __init__(
        self,
        vocab: Vocabulary,
        seq2vec_encoder: Seq2VecEncoder,
        feedforward: Optional[FeedForward] = None,
        input_dim: int = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if self._feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = (
                self._seq2vec_encoder.get_output_dim() or input_dim
            )

        if self._classifier_input_dim is None:
            raise ValueError("No input dimension given!")

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.size(namespace=self._label_namespace)
        self._classification_layer = torch.nn.Linear(
            self._classifier_input_dim, self._num_labels
        )
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(  # type: ignore
        self,
        inputs: torch.FloatTensor,
        mask: torch.BoolTensor,
        label: torch.IntTensor = None,
    ) -> Dict[str, torch.Tensor]:
        encoding = self._seq2vec_encoder(inputs, mask=mask)

        if self._dropout:
            encoding = self._dropout(encoding)

        if self._feedforward is not None:
            encoding = self._feedforward(encoding)

        logits = self._classification_layer(encoding)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss

        return output_dict

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        if "probs" in output_dict:
            predictions = output_dict["probs"]
            if predictions.dim() == 2:
                predictions_list = [predictions[i] for i in range(predictions.shape[0])]
            else:
                predictions_list = [predictions]
            classes = []
            for prediction in predictions_list:
                label_idx = prediction.argmax(dim=-1).item()
                label_str = self.vocab.get_index_to_token_vocabulary(
                    self._label_namespace
                ).get(label_idx, str(label_idx))
                classes.append(label_str)
            output_dict["label"] = classes
        return output_dict
