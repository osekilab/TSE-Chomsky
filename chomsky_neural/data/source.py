from typing import Any, Iterator, List

from chomsky_neural.data.instance import Pair, Sentence
from chomsky_neural.env import DATA_DIR
from chomsky_neural.file_utils import load_jsonlines


class DataSource:
    def collect(self) -> Iterator[Any]:
        raise NotImplementedError


class ChomskyDataSource(DataSource):
    def __init__(self, dataname: str, subset: str, correct_only: bool = False) -> None:
        self.dataname = dataname
        self.subset = subset
        self.correct_only = correct_only
        if self.correct_only:
            self._datasource = [
                data
                for data in load_jsonlines(DATA_DIR / f"{dataname}/{subset}.jsonl")
                if data["labels"] == 1
            ]
        else:
            self._datasource = load_jsonlines(DATA_DIR / f"{dataname}/{subset}.jsonl")
        self._data: List[Sentence] = []
        self._build_data()

    def _build_data(self) -> None:
        for data in self._datasource:
            sentence = Sentence(
                sequences=data["sequences"],
                label=data["labels"],
                data_type=data["data_type"],
                num_seq=data["num_seq"],
            )
            if self.correct_only:
                assert sentence.label == 1
            self._data.append(sentence)

    def collect(self) -> Iterator[Any]:
        yield from self._data

    def get_features(self, feature: str) -> List[Any]:
        return [getattr(sentence, feature) for sentence in self._data]


class ChomskyPairDataSource(DataSource):
    def __init__(self, dataname: str, subset: str) -> None:
        self.dataname = dataname
        self.subset = subset
        self._datasource = load_jsonlines(DATA_DIR / f"{dataname}/{subset}.jsonl")
        self._data: List[Pair] = []
        self._build_data()

    def _build_data(self) -> None:
        for data in self._datasource:
            pair = Pair(
                sequences_pos=data["good_sequences"],
                sequences_neg=data["bad_sequences"],
                num_seq_pos=data["num_good_seq"],
                num_seq_neg=data["num_bad_seq"],
            )
            self._data.append(pair)

    def collect(self) -> Iterator[Any]:
        yield from self._data

    def get_features(self, feature: str) -> List[Any]:
        return [getattr(pair, feature) for pair in self._data]
