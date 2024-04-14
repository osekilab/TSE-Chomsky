from pathlib import Path
from typing import Any, Dict, List, Union, cast

import numpy as np
from chomsky_neural.file_utils import save_as_jsonlines


def zipf_sample(data: List, size: int, replace: bool = True) -> List[int]:
    """Zipfian sampling."""
    assert size <= len(data) or replace
    return np.random.choice(
        data, size=size, replace=replace, p=zipf_prob(data)
    ).tolist()


def zipf_prob(data: List) -> list:
    """Zipfian probability."""
    freq = np.array([1 / (i + 1) for i in range(len(data))])
    return freq / sum(freq)


def sample_from_list(
    data: List, size: int, replace: bool = True, dist: str = "uniform"
) -> List[int]:
    """Sample from list."""
    if dist == "uniform":
        return np.random.choice(data, size=size, replace=replace).tolist()
    elif dist == "zipf":
        return zipf_sample(data, size=size, replace=replace)
    else:
        raise ValueError(f"Unknown dist: {dist}")


def save_split_data(
    train_data: List[Dict],
    validation_data: List[Dict],
    test_data: List[Dict],
    out_of_dist_data: List[Dict],
    output_dir: Union[str, Path],
) -> None:
    output_dir = Path(output_dir)
    save_as_jsonlines(train_data, output_dir / "train.jsonl")
    save_as_jsonlines(validation_data, output_dir / "validation.jsonl")
    save_as_jsonlines(test_data, output_dir / "test.jsonl")
    save_as_jsonlines(out_of_dist_data, output_dir / "test_out_of_dist.jsonl")


def get_unique_list(data: List[List[str]]) -> List[List[str]]:
    """Get unique list."""
    return list(map(list, set(map(tuple, data))))  # type: ignore


def create_sample(
    sequences: List[str],
    labels: int,
    data_type: str = "correct",
) -> Dict[str, Any]:
    """
    データの1事例を生成する関数

    Arguments:
        sequences -- 入力系列
        labels -- 正解ラベル

    Keyword Arguments:
        data_type -- データの生成方法について。(default: {'correct'})
            正例であれば'correct'とし、負例であればその生成方法を入れる。
            e.g.) flip, wrong_num_iterations, wrong_dependency ...
    """
    return {
        "sequences": sequences,
        "labels": labels,
        "data_type": data_type,
        "num_seq": len(sequences),
    }


def create_samples(
    sequences: List[List[str]],
    labels: int,
    data_type: str = "correct",
) -> List[Dict[str, Any]]:
    samples = []
    for seq in sequences:
        sample = {
            "sequences": seq,
            "labels": labels,
            "data_type": data_type,
            "num_seq": len(seq),
        }
        samples.append(sample)

    return samples


def create_pair_samples(
    good_sequences: List[List[str]],
    bad_sequences: List[List[str]],
) -> List[Dict[str, Any]]:
    samples = []
    for good_seq, bad_seq in zip(good_sequences, bad_sequences):
        sample = {
            "good_sequences": good_seq,
            "bad_sequences": bad_seq,
            "num_good_seq": len(good_seq),
            "num_bad_seq": len(bad_seq),
        }
        samples.append(sample)

    return samples


def generate_vocab_file(output_dir: Path, vocab_size: int) -> None:
    vocab_file_path = output_dir / "vocab.txt"
    with open(vocab_file_path, "w") as f:
        for i in range(vocab_size):
            f.write(f"{str(i)}\n")
        f.write("[PAD]\n")
        f.write("[CLS]\n")
        f.write("[SEP]\n")
