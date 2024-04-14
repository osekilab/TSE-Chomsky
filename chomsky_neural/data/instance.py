from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Batch:
    inputs: torch.LongTensor  # (batch_size, sequence_length)
    mask: torch.BoolTensor  # (batch_size, sequence_length)
    label: Optional[torch.IntTensor] = None  # (batch_size,)


@dataclass
class Sentence:
    sequences: List[str]
    label: int
    data_type: str
    num_seq: int


@dataclass
class Instance:
    inputs: List[int]
    label: Optional[int] = None


@dataclass
class PairBatch:
    inputs_pos: torch.LongTensor  # (batch_size, sequence_length)
    inputs_neg: torch.LongTensor
    mask_pos: torch.BoolTensor
    mask_neg: torch.BoolTensor


@dataclass
class Pair:
    sequences_pos: List[str]
    sequences_neg: List[str]
    num_seq_pos: int
    num_seq_neg: int


@dataclass
class PairInstance:
    inputs_pos: List[int]
    inputs_neg: List[int]
