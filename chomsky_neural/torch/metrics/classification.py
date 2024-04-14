from typing import Literal, Optional

from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


def get_classification_full_metrics(
    task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average_method: Literal["micro", "macro", "weighted", "none"] = "micro",
    multidim_average: Literal["global", "samplewise"] = "global",
    top_k: Optional[int] = 1,
    ignore_index: Optional[int] = None,
    validate_args: bool = True,
    prefix: Optional[str] = None,
    kwargs: dict = {},
) -> MetricCollection:
    return MetricCollection(
        [
            Accuracy(
                task=task,
                threshold=threshold,
                num_classes=num_classes,
                num_labels=num_labels,
                average=average_method,
                multidim_average=multidim_average,
                top_k=top_k,
                ignore_index=ignore_index,
                validate_args=validate_args,
                **kwargs,
            ),
            Precision(
                task=task,
                threshold=threshold,
                num_classes=num_classes,
                num_labels=num_labels,
                average=average_method,
                multidim_average=multidim_average,
                top_k=top_k,
                ignore_index=ignore_index,
                validate_args=validate_args,
                **kwargs,
            ),
            Recall(
                task=task,
                threshold=threshold,
                num_classes=num_classes,
                num_labels=num_labels,
                average=average_method,
                multidim_average=multidim_average,
                top_k=top_k,
                ignore_index=ignore_index,
                validate_args=validate_args,
                **kwargs,
            ),
            F1Score(
                task=task,
                threshold=threshold,
                num_classes=num_classes,
                num_labels=num_labels,
                average=average_method,
                multidim_average=multidim_average,
                top_k=top_k,
                ignore_index=ignore_index,
                validate_args=validate_args,
                **kwargs,
            ),
        ],
        prefix=prefix,
    )
