from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from chomsky_neural.data.tasks import (
    generate_AnBm,
    generate_cross_serial_dependency,
    generate_nested_dependency,
)
from chomsky_neural.data.utils import (
    create_pair_samples,
    create_samples,
    generate_vocab_file,
    get_unique_list,
)
from chomsky_neural.env import PACKAGE_DIR
from chomsky_neural.file_utils import save_as_jsonlines
from chomsky_neural.data.utils import sample_from_list
from sklearn.model_selection import train_test_split

np.random.seed(42)

IN_DIST_TRAIN_SIZE = 900
IN_DIST_VAL_SIZE = 100
IN_DIST_TEST_SIZE = 100
IN_DIST_PAIR_SIZE = 100
OUT_OF_DIST_TEST_SIZE = 100
OUT_OF_DIST_PAIR_SIZE = 100


def generate_AnB_data(
    num_of_terms: int = 1,
    min_iter: int = 3,
    max_iter: int = 29,
    max_iter_test: int = 99,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> None:
    """
    Generates AnB data.

    Keyword Arguments:
        num_of_terms -- number of terminals for each nonterminal (default: {1})
        min_iter -- minimum number of iterations (=n) (default: {2})
        max_iter -- maximum number of iterations (=n) in train/dev data (default: {10})
        max_iter_test -- maximum number of iterations (=n) in test data (default: {50})
        in_dist_data_size -- number of in-distribution data
            if this is too large, duplicated data will be erased. (default: {10000})
    """
    task_name = f"AnB_{num_of_terms}terms_{min_iter}_{max_iter}_{max_iter_test}_{'erase-dup' if erase_duplicated_data else 'keep-dup'}_{dist}"
    data_dir = PACKAGE_DIR / "data"
    output_dir = data_dir / task_name

    # generate samples
    in_dist_positive = generate_AnB_positive_samples(
        IN_DIST_TRAIN_SIZE + IN_DIST_VAL_SIZE + IN_DIST_TEST_SIZE + IN_DIST_PAIR_SIZE,
        num_of_terms,
        min_iter=min_iter,
        max_iter=max_iter,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    # split
    in_dist_train_val, in_dist_test_pair = train_test_split(
        in_dist_positive,
        test_size=IN_DIST_TEST_SIZE + IN_DIST_PAIR_SIZE,
    )
    in_dist_train, in_dist_val = train_test_split(
        in_dist_train_val,
        test_size=IN_DIST_VAL_SIZE,
    )
    in_dist_test_positive, in_dist_pair_positive = train_test_split(
        in_dist_test_pair,
        test_size=IN_DIST_PAIR_SIZE,
    )
    in_dist_pair_positive_train, in_dist_pair_positive_val = train_test_split(
        in_dist_pair_positive,
        test_size=0.1,
    )
    in_dist_pair_negative_train = generate_AnB_negative_samples(
        in_dist_pair_positive_train,
        num_of_terms,
        dist=dist,
    )
    in_dist_pair_negative_val = generate_AnB_negative_samples(
        in_dist_pair_positive_val,
        num_of_terms,
        dist=dist,
    )
    in_dist_test_negative = generate_AnB_negative_samples(
        in_dist_test_positive,
        num_of_terms,
        dist=dist,
    )
    in_dist_train_samples = create_samples(in_dist_train, labels=1, data_type="correct")
    in_dist_val_samples = create_samples(in_dist_val, labels=1, data_type="correct")
    in_dist_test_positive_samples = create_samples(
        in_dist_test_positive, labels=1, data_type="correct"
    )
    in_dist_test_negative_samples = create_samples(
        in_dist_test_negative, labels=0, data_type="flip"
    )
    in_dist_test_samples = in_dist_test_positive_samples + in_dist_test_negative_samples
    in_dist_test_pair_samples = create_pair_samples(
        in_dist_test_positive, in_dist_test_negative
    )
    in_dist_train_pair_samples = create_samples(
        in_dist_pair_positive_train, labels=1, data_type="correct"
    ) + create_samples(in_dist_pair_negative_train, labels=0, data_type="flip")
    in_dist_val_pair_samples = create_samples(
        in_dist_pair_positive_val, labels=1, data_type="correct"
    ) + create_samples(in_dist_pair_negative_val, labels=0, data_type="flip")

    # output
    output_dir = Path(output_dir)
    save_as_jsonlines(in_dist_train_samples, output_dir / "train.jsonl")
    save_as_jsonlines(in_dist_val_samples, output_dir / "validation.jsonl")
    save_as_jsonlines(in_dist_test_samples, output_dir / "test.jsonl")
    save_as_jsonlines(in_dist_train_pair_samples, output_dir / "train_mlp.jsonl")
    save_as_jsonlines(in_dist_val_pair_samples, output_dir / "validation_mlp.jsonl")
    save_as_jsonlines(in_dist_test_pair_samples, output_dir / "test_pair.jsonl")

    # out_dist
    out_of_dist_positive = generate_AnB_positive_samples(
        OUT_OF_DIST_TEST_SIZE + OUT_OF_DIST_PAIR_SIZE,
        num_of_terms,
        min_iter=max_iter + 1,
        max_iter=max_iter_test,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    out_of_dist_pair_positive, out_of_dist_test_positive = train_test_split(
        out_of_dist_positive,
        test_size=OUT_OF_DIST_TEST_SIZE,
    )
    out_of_dist_pair_positive_train, out_of_dist_pair_positive_val = train_test_split(
        out_of_dist_pair_positive,
        test_size=0.1,
    )
    out_of_dist_pair_negative_train = generate_AnB_negative_samples(
        out_of_dist_pair_positive_train,
        num_of_terms,
        dist=dist,
    )
    out_of_dist_pair_negative_val = generate_AnB_negative_samples(
        out_of_dist_pair_positive_val,
        num_of_terms,
        dist=dist,
    )
    out_of_dist_test_negative = generate_AnB_negative_samples(
        out_of_dist_test_positive,
        num_of_terms,
        dist=dist,
    )

    # create sample
    out_of_dist_test_positive_samples = create_samples(
        out_of_dist_test_positive, labels=1, data_type="correct"
    )
    out_of_dist_test_negative_samples = create_samples(
        out_of_dist_test_negative, labels=0, data_type="flip"
    )
    out_of_dist_test_samples = (
        out_of_dist_test_positive_samples + out_of_dist_test_negative_samples
    )
    out_of_dist_pair_test_samples = create_pair_samples(
        out_of_dist_test_positive, out_of_dist_test_negative
    )
    out_of_dist_train_positive_samples = create_samples(
        out_of_dist_pair_positive_train, labels=1, data_type="correct"
    )
    out_of_dist_train_negative_samples = create_samples(
        out_of_dist_pair_negative_train, labels=0, data_type="flip"
    )
    out_of_dist_train_samples = (
        out_of_dist_train_positive_samples + out_of_dist_train_negative_samples
    )
    out_of_dist_val_positive_samples = create_samples(
        out_of_dist_pair_positive_val, labels=1, data_type="correct"
    )
    out_of_dist_val_negative_samples = create_samples(
        out_of_dist_pair_negative_val, labels=0, data_type="flip"
    )
    out_of_dist_val_samples = (
        out_of_dist_val_positive_samples + out_of_dist_val_negative_samples
    )
    # output
    save_as_jsonlines(out_of_dist_train_samples, output_dir / "out_of_dist_train.jsonl")
    save_as_jsonlines(out_of_dist_val_samples, output_dir / "out_of_dist_val.jsonl")
    save_as_jsonlines(out_of_dist_test_samples, output_dir / "out_of_dist_test.jsonl")
    save_as_jsonlines(
        out_of_dist_pair_test_samples, output_dir / "out_of_dist_test_pair.jsonl"
    )
    generate_vocab_file(output_dir, num_of_terms * 2)


def generate_AnB_positive_samples(
    data_size: int,
    num_of_terms: int,
    min_iter: int,
    max_iter: int,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> List[List[str]]:
    positive: List[List[str]] = []
    while len(positive) < data_size:
        num_iter = np.random.randint(min_iter, max_iter + 1)
        sample = generate_AnBm(num_of_terms, num_of_As=num_iter, num_of_Bs=1, dist=dist)
        positive.append(sample)
        if erase_duplicated_data:
            positive = get_unique_list(positive)
        if len(positive) % 1000 == 0:
            print(f"generated {len(positive)} positive examples...")
    assert len(positive) == data_size, f"{len(positive)} != data_size (={data_size})"

    return positive


def generate_AnB_negative_samples(
    positive_samples: List[List[str]],
    num_of_terms: int,
    dist: str = "uniform",
) -> List[List[str]]:
    negative_samples: List[List[str]] = []
    for positive_sample in positive_samples:
        num_iter = len(positive_sample) - 1
        num_flip = np.random.randint(1, num_iter + 1)
        flip_sites = np.random.choice(np.arange(num_iter), size=num_flip, replace=False)
        term_B = list(range(num_of_terms, 2 * num_of_terms, 1))
        negative_sample = positive_sample.copy()
        for flip_site in flip_sites:
            negative_sample[flip_site] = str(
                sample_from_list(term_B, size=1, replace=True, dist=dist)[0]
            )
        negative_samples.append(negative_sample)
        assert len(negative_sample) == len(positive_sample)
        if len(negative_samples) % 1000 == 0:
            print(f"generated {len(negative_samples)} negative examples...")
    assert len(negative_samples) == len(
        positive_samples
    ), f"{len(negative_samples)} != len(positive_samples) (={len(positive_samples)})"
    return negative_samples


def generate_AnB_samples(
    data_size: int,
    num_of_terms: int,
    min_iter: int,
    max_iter: int,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> Tuple[List[List[str]], List[List[str]]]:
    assert data_size % 2 == 0, "data_size must be even."
    positive = generate_AnB_positive_samples(
        int(data_size / 2),
        num_of_terms,
        min_iter=min_iter,
        max_iter=max_iter,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    negative = generate_AnB_negative_samples(
        positive,
        num_of_terms,
        dist=dist,
    )
    return positive, negative


def generate_AnBn_data(
    num_of_terms: int = 1,
    min_iter: int = 2,
    max_iter: int = 15,
    max_iter_test: int = 50,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> None:
    """
    Generates AnBn data.

    Keyword Arguments:
        num_of_terms -- number of terminals for each nonterminal (default: {1})
        min_iter -- minimum number of iterations (=n) (default: {2})
        max_iter -- maximum number of iterations (=n) in train/dev data (default: {10})
        max_iter_test -- maximum number of iterations (=n) in test data (default: {50})
        in_dist_data_size -- number of in-distribution data
            if this is too large, duplicated data will be erased. (default: {10000})
    """
    task_name = f"AnBn_{num_of_terms}terms_{min_iter}_{max_iter}_{max_iter_test}_{'erase-dup' if erase_duplicated_data else 'keep-dup'}_{dist}"
    data_dir = PACKAGE_DIR / "data"
    output_dir = data_dir / task_name

    # generate samples
    in_dist_positive = generate_AnBn_positive_samples(
        IN_DIST_TRAIN_SIZE + IN_DIST_VAL_SIZE + IN_DIST_TEST_SIZE + IN_DIST_PAIR_SIZE,
        num_of_terms,
        min_iter=min_iter,
        max_iter=max_iter,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    # split
    in_dist_train_val, in_dist_test_pair = train_test_split(
        in_dist_positive,
        test_size=IN_DIST_TEST_SIZE + IN_DIST_PAIR_SIZE,
    )
    in_dist_train, in_dist_val = train_test_split(
        in_dist_train_val,
        test_size=IN_DIST_VAL_SIZE,
    )
    in_dist_test_positive, in_dist_pair_positive = train_test_split(
        in_dist_test_pair,
        test_size=IN_DIST_PAIR_SIZE,
    )
    in_dist_pair_positive_train, in_dist_pair_positive_val = train_test_split(
        in_dist_pair_positive,
        test_size=0.1,
    )
    in_dist_pair_negative_train = generate_AnBn_negative_samples(
        in_dist_pair_positive_train,
        num_of_terms,
        dist=dist,
    )
    in_dist_pair_negative_val = generate_AnBn_negative_samples(
        in_dist_pair_positive_val,
        num_of_terms,
        dist=dist,
    )
    in_dist_test_negative = generate_AnBn_negative_samples(
        in_dist_test_positive,
        num_of_terms,
        dist=dist,
    )
    in_dist_train_samples = create_samples(in_dist_train, labels=1, data_type="correct")
    in_dist_val_samples = create_samples(in_dist_val, labels=1, data_type="correct")
    in_dist_test_positive_samples = create_samples(
        in_dist_test_positive, labels=1, data_type="correct"
    )
    in_dist_test_negative_samples = create_samples(
        in_dist_test_negative, labels=0, data_type="flip"
    )
    in_dist_test_samples = in_dist_test_positive_samples + in_dist_test_negative_samples
    in_dist_test_pair_samples = create_pair_samples(
        in_dist_test_positive, in_dist_test_negative
    )
    in_dist_train_pair_samples = create_samples(
        in_dist_pair_positive_train, labels=1, data_type="correct"
    ) + create_samples(in_dist_pair_negative_train, labels=0, data_type="flip")
    in_dist_val_pair_samples = create_samples(
        in_dist_pair_positive_val, labels=1, data_type="correct"
    ) + create_samples(in_dist_pair_negative_val, labels=0, data_type="flip")

    # output
    output_dir = Path(output_dir)
    save_as_jsonlines(in_dist_train_samples, output_dir / "train.jsonl")
    save_as_jsonlines(in_dist_val_samples, output_dir / "validation.jsonl")
    save_as_jsonlines(in_dist_test_samples, output_dir / "test.jsonl")
    save_as_jsonlines(in_dist_train_pair_samples, output_dir / "train_mlp.jsonl")
    save_as_jsonlines(in_dist_val_pair_samples, output_dir / "validation_mlp.jsonl")
    save_as_jsonlines(in_dist_test_pair_samples, output_dir / "test_pair.jsonl")

    # out_dist
    out_of_dist_positive = generate_AnBn_positive_samples(
        OUT_OF_DIST_TEST_SIZE + OUT_OF_DIST_PAIR_SIZE,
        num_of_terms,
        min_iter=max_iter + 1,
        max_iter=max_iter_test,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    out_of_dist_pair_positive, out_of_dist_test_positive = train_test_split(
        out_of_dist_positive,
        test_size=OUT_OF_DIST_TEST_SIZE,
    )
    out_of_dist_pair_positive_train, out_of_dist_pair_positive_val = train_test_split(
        out_of_dist_pair_positive,
        test_size=0.1,
    )
    out_of_dist_pair_negative_train = generate_AnBn_negative_samples(
        out_of_dist_pair_positive_train,
        num_of_terms,
        dist=dist,
    )
    out_of_dist_pair_negative_val = generate_AnBn_negative_samples(
        out_of_dist_pair_positive_val,
        num_of_terms,
        dist=dist,
    )
    out_of_dist_test_negative = generate_AnBn_negative_samples(
        out_of_dist_test_positive,
        num_of_terms,
        dist=dist,
    )

    # create sample
    out_of_dist_test_positive_samples = create_samples(
        out_of_dist_test_positive, labels=1, data_type="correct"
    )
    out_of_dist_test_negative_samples = create_samples(
        out_of_dist_test_negative, labels=0, data_type="flip"
    )
    out_of_dist_test_samples = (
        out_of_dist_test_positive_samples + out_of_dist_test_negative_samples
    )
    out_of_dist_pair_test_samples = create_pair_samples(
        out_of_dist_test_positive, out_of_dist_test_negative
    )
    out_of_dist_train_positive_samples = create_samples(
        out_of_dist_pair_positive_train, labels=1, data_type="correct"
    )
    out_of_dist_train_negative_samples = create_samples(
        out_of_dist_pair_negative_train, labels=0, data_type="flip"
    )
    out_of_dist_train_samples = (
        out_of_dist_train_positive_samples + out_of_dist_train_negative_samples
    )
    out_of_dist_val_positive_samples = create_samples(
        out_of_dist_pair_positive_val, labels=1, data_type="correct"
    )
    out_of_dist_val_negative_samples = create_samples(
        out_of_dist_pair_negative_val, labels=0, data_type="flip"
    )
    out_of_dist_val_samples = (
        out_of_dist_val_positive_samples + out_of_dist_val_negative_samples
    )
    # output
    save_as_jsonlines(out_of_dist_train_samples, output_dir / "out_of_dist_train.jsonl")
    save_as_jsonlines(out_of_dist_val_samples, output_dir / "out_of_dist_val.jsonl")
    save_as_jsonlines(out_of_dist_test_samples, output_dir / "out_of_dist_test.jsonl")
    save_as_jsonlines(
        out_of_dist_pair_test_samples, output_dir / "out_of_dist_test_pair.jsonl"
    )
    generate_vocab_file(output_dir, num_of_terms * 2)


def generate_AnBn_positive_samples(
    data_size: int,
    num_of_terms: int,
    min_iter: int,
    max_iter: int,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> List[List[str]]:
    positive: List[List[str]] = []
    while len(positive) < data_size:
        num_iter = np.random.randint(min_iter, max_iter + 1)
        sample = generate_AnBm(num_of_terms, num_iter, num_iter, dist=dist)
        positive.append(sample)
        if erase_duplicated_data:
            positive = get_unique_list(positive)
        if len(positive) % 1000 == 0:
            print(f"generated {len(positive)} positive examples...")
    assert len(positive) == data_size, f"{len(positive)} != data_size (={data_size})"
    return positive


def generate_AnBn_negative_samples(
    positive_samples: List[List[str]],
    num_of_terms: int,
    dist: str = "uniform",
) -> List[List[str]]:
    negative_samples: List[List[str]] = []
    term_A = list(range(0, num_of_terms, 1))
    term_B = list(range(num_of_terms, 2 * num_of_terms, 1))
    for positive_sample in positive_samples:
        assert len(positive_sample) % 2 == 0, f"{len(positive_sample)} is not even."
        negative_example = positive_sample.copy()
        add_or_subtract = np.random.choice([-1, 1])
        num_a_or_b = np.random.choice(["a", "b"])
        if num_a_or_b == "a":
            insert_or_delete_site = np.random.randint(0, int(len(positive_sample) / 2))
            if add_or_subtract == 1:
                # insert one extra A (A is from 0 to len(positive_sample) / 2)
                negative_example.insert(
                    insert_or_delete_site,
                    str(sample_from_list(term_A, size=1, replace=True, dist=dist)[0]),
                )
                assert len(negative_example) == len(positive_sample) + 1
                for i in range(int(len(positive_sample) / 2) + 1):
                    assert int(negative_example[i]) in term_A
                for i in range(
                    int(len(positive_sample) / 2) + 1, len(positive_sample) + 1
                ):
                    assert int(negative_example[i]) in term_B

            else:  # add_or_subtract == -1
                # delete one A
                negative_example.pop(insert_or_delete_site)
                assert len(negative_example) == len(positive_sample) - 1
                for i in range(int(len(positive_sample) / 2) - 1):
                    assert int(negative_example[i]) in term_A
                for i in range(
                    int(len(positive_sample) / 2) - 1, len(positive_sample) - 1
                ):
                    assert int(negative_example[i]) in term_B
        else:
            insert_or_delete_site = np.random.randint(
                int(len(positive_sample) / 2), len(positive_sample)
            )
            if add_or_subtract == 1:
                # insert one extra B (B is from len(positive_sample) / 2 to len(positive_sample))
                negative_example.insert(
                    insert_or_delete_site,
                    str(sample_from_list(term_B, size=1, replace=True, dist=dist)[0]),
                )
                assert len(negative_example) == len(positive_sample) + 1
                for i in range(int(len(positive_sample) / 2)):
                    assert int(negative_example[i]) in term_A, f"{negative_example}"
                for i in range(int(len(positive_sample) / 2), len(positive_sample) + 1):
                    assert int(negative_example[i]) in term_B, f"{negative_example}"
            else:
                # delete one B
                negative_example.pop(insert_or_delete_site)
                assert len(negative_example) == len(positive_sample) - 1
                for i in range(int(len(positive_sample) / 2)):
                    assert int(negative_example[i]) in term_A
                for i in range(int(len(positive_sample) / 2), len(positive_sample) - 1):
                    assert int(negative_example[i]) in term_B, f"{negative_example}"
        negative_samples.append(negative_example)
        if len(negative_samples) % 1000 == 0:
            print(f"generated {len(negative_samples)} negative examples...")
    assert len(negative_samples) == len(
        positive_samples
    ), f"{len(negative_samples)} != len(positive_samples) (={len(positive_samples)})"
    return negative_samples


def generate_AnBn_samples(
    data_size: int,
    num_of_terms: int,
    min_iter: int,
    max_iter: int,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> Tuple[List[List[str]], List[List[str]]]:
    assert data_size % 2 == 0, "data_size must be even."
    positive = generate_AnBn_positive_samples(
        int(data_size / 2),
        num_of_terms,
        min_iter=min_iter,
        max_iter=max_iter,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    negative = generate_AnBn_negative_samples(
        positive,
        num_of_terms,
        dist=dist,
    )
    return positive, negative


def generate_dependency_data(
    dependency_type: str,
    num_of_terms: int = 2,
    num_of_nonterms: int = 5,
    min_dep: int = 2,
    max_dep: int = 15,
    max_dep_test: int = 50,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> None:
    """
    Generate dependency data.

    Arguments:
        dependency_type -- type of dependency (cross_serial or nested)

    Keyword Arguments:
        num_of_terms -- number of terminals (default: {2})
        num_of_nonterms -- number of nonterminals (default: {2})
        min_dep -- minimum number of dependencies (default: {2})
        max_dep -- maximum number of dependencies for train/dev data (default: {10})
        max_dep_test -- maximum number of dependencies for test data (default: {50})
        erase_duplicated_data -- erase duplicated data or not (default: {True})
    """
    task_name = f"{dependency_type}_{num_of_terms}terms_{num_of_nonterms}nonterms_{min_dep}_{max_dep}_{max_dep_test}_{'erase-dup' if erase_duplicated_data else 'keep-dup'}_{dist}"
    data_dir = PACKAGE_DIR / "data"
    output_dir = data_dir / task_name

    # generate samples
    in_dist_positive = generate_dependency_positive_samples(
        IN_DIST_TRAIN_SIZE + IN_DIST_VAL_SIZE + IN_DIST_TEST_SIZE + IN_DIST_PAIR_SIZE,
        num_of_terms,
        num_of_nonterms=num_of_nonterms,
        min_dep=min_dep,
        max_dep=max_dep,
        dependency_type=dependency_type,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    # split
    in_dist_train_val, in_dist_test_pair = train_test_split(
        in_dist_positive,
        test_size=IN_DIST_TEST_SIZE + IN_DIST_PAIR_SIZE,
    )
    in_dist_train, in_dist_val = train_test_split(
        in_dist_train_val,
        test_size=IN_DIST_VAL_SIZE,
    )
    in_dist_test_positive, in_dist_pair_positive = train_test_split(
        in_dist_test_pair,
        test_size=IN_DIST_PAIR_SIZE,
    )
    in_dist_pair_positive_train, in_dist_pair_positive_val = train_test_split(
        in_dist_pair_positive,
        test_size=0.1,
    )
    in_dist_pair_negative_train = generate_dependency_negative_samples(
        in_dist_pair_positive_train
    )
    in_dist_pair_negative_val = generate_dependency_negative_samples(
        in_dist_pair_positive_val
    )
    in_dist_test_negative = generate_dependency_negative_samples(in_dist_test_positive)
    in_dist_train_samples = create_samples(in_dist_train, labels=1, data_type="correct")
    in_dist_val_samples = create_samples(in_dist_val, labels=1, data_type="correct")
    in_dist_test_positive_samples = create_samples(
        in_dist_test_positive, labels=1, data_type="correct"
    )
    in_dist_test_negative_samples = create_samples(
        in_dist_test_negative, labels=0, data_type="flip"
    )
    in_dist_test_samples = in_dist_test_positive_samples + in_dist_test_negative_samples
    in_dist_test_pair_samples = create_pair_samples(
        in_dist_test_positive, in_dist_test_negative
    )
    in_dist_train_pair_samples = create_samples(
        in_dist_pair_positive_train, labels=1, data_type="correct"
    ) + create_samples(in_dist_pair_negative_train, labels=0, data_type="flip")
    in_dist_val_pair_samples = create_samples(
        in_dist_pair_positive_val, labels=1, data_type="correct"
    ) + create_samples(in_dist_pair_negative_val, labels=0, data_type="flip")

    # output
    output_dir = Path(output_dir)
    save_as_jsonlines(in_dist_train_samples, output_dir / "train.jsonl")
    save_as_jsonlines(in_dist_val_samples, output_dir / "validation.jsonl")
    save_as_jsonlines(in_dist_test_samples, output_dir / "test.jsonl")
    save_as_jsonlines(in_dist_train_pair_samples, output_dir / "train_mlp.jsonl")
    save_as_jsonlines(in_dist_val_pair_samples, output_dir / "validation_mlp.jsonl")
    save_as_jsonlines(in_dist_test_pair_samples, output_dir / "test_pair.jsonl")

    # out_dist
    out_of_dist_positive = generate_dependency_positive_samples(
        OUT_OF_DIST_TEST_SIZE + OUT_OF_DIST_PAIR_SIZE,
        num_of_terms,
        num_of_nonterms=num_of_nonterms,
        min_dep=max_dep + 1,
        max_dep=max_dep_test,
        dependency_type=dependency_type,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    out_of_dist_pair_positive, out_of_dist_test_positive = train_test_split(
        out_of_dist_positive,
        test_size=OUT_OF_DIST_TEST_SIZE,
    )
    out_of_dist_pair_positive_train, out_of_dist_pair_positive_val = train_test_split(
        out_of_dist_pair_positive,
        test_size=0.1,
    )
    out_of_dist_pair_negative_train = generate_dependency_negative_samples(
        out_of_dist_pair_positive_train,
    )
    out_of_dist_pair_negative_val = generate_dependency_negative_samples(
        out_of_dist_pair_positive_val,
    )
    out_of_dist_test_negative = generate_dependency_negative_samples(
        out_of_dist_test_positive,
    )

    # create sample
    out_of_dist_test_positive_samples = create_samples(
        out_of_dist_test_positive, labels=1, data_type="correct"
    )
    out_of_dist_test_negative_samples = create_samples(
        out_of_dist_test_negative, labels=0, data_type="flip"
    )
    out_of_dist_test_samples = (
        out_of_dist_test_positive_samples + out_of_dist_test_negative_samples
    )
    out_of_dist_pair_test_samples = create_pair_samples(
        out_of_dist_test_positive, out_of_dist_test_negative
    )
    out_of_dist_train_positive_samples = create_samples(
        out_of_dist_pair_positive_train, labels=1, data_type="correct"
    )
    out_of_dist_train_negative_samples = create_samples(
        out_of_dist_pair_negative_train, labels=0, data_type="flip"
    )
    out_of_dist_train_samples = (
        out_of_dist_train_positive_samples + out_of_dist_train_negative_samples
    )
    out_of_dist_val_positive_samples = create_samples(
        out_of_dist_pair_positive_val, labels=1, data_type="correct"
    )
    out_of_dist_val_negative_samples = create_samples(
        out_of_dist_pair_negative_val, labels=0, data_type="flip"
    )
    out_of_dist_val_samples = (
        out_of_dist_val_positive_samples + out_of_dist_val_negative_samples
    )
    # output
    save_as_jsonlines(out_of_dist_train_samples, output_dir / "out_of_dist_train.jsonl")
    save_as_jsonlines(out_of_dist_val_samples, output_dir / "out_of_dist_val.jsonl")
    save_as_jsonlines(out_of_dist_test_samples, output_dir / "out_of_dist_test.jsonl")
    save_as_jsonlines(
        out_of_dist_pair_test_samples, output_dir / "out_of_dist_test_pair.jsonl"
    )
    generate_vocab_file(output_dir, num_of_terms * num_of_nonterms * 2)


def generate_dependency_positive_samples(
    data_size: int,
    num_of_terms: int,
    num_of_nonterms: int,
    min_dep: int,
    max_dep: int,
    dependency_type: str,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> List[List[str]]:
    # choose generation func.
    if dependency_type == "cross_serial":
        generate_func = generate_cross_serial_dependency
    elif dependency_type == "nested":
        generate_func = generate_nested_dependency
    else:
        raise ValueError('dependency type must be either "cross_serial" or "nested"')

    positive: List[List[str]] = []
    while len(positive) < data_size:
        num_of_dependencies = np.random.randint(min_dep, max_dep + 1)
        sample = generate_func(
            num_of_nonterms, num_of_terms, num_of_dependencies, 0, dist=dist
        )
        positive.append(sample)
        if erase_duplicated_data:
            positive = get_unique_list(positive)
        if len(positive) % 1000 == 0:
            print(f"generated {len(positive)} positive examples...")
    assert len(positive) == data_size, f"{len(positive)} != data_size (={data_size})"

    return positive


def generate_dependency_negative_samples(
    positive_samples: List[List[str]],
) -> List[List[str]]:
    negative_samples: List[List[str]] = []
    for positive_sample in positive_samples:
        assert len(positive_sample) % 2 == 0, f"{len(positive_sample)} is not even."
        negative_sample = positive_sample.copy()
        flip_a_or_b = np.random.choice(["a", "b"])
        if flip_a_or_b == "a":
            flip_sites = np.random.choice(
                np.arange(0, int(len(positive_sample) / 2)),
                size=2,
                replace=False,
            )
            # flip negative_sample[flip_sites[0]] and negative_sample[flip_sites[1]]
            negative_sample[flip_sites[0]], negative_sample[flip_sites[1]] = (
                negative_sample[flip_sites[1]],
                negative_sample[flip_sites[0]],
            )
            assert len(negative_sample) == len(positive_sample)
            assert set(negative_sample[: int(len(positive_sample) / 2)]) == set(
                positive_sample[: int(len(positive_sample) / 2)]
            )
            assert (
                negative_sample[int(len(positive_sample) / 2) :]
                == positive_sample[int(len(positive_sample) / 2) :]
            )

        else:
            flip_sites = np.random.choice(
                np.arange(int(len(positive_sample) / 2), len(positive_sample)),
                size=2,
                replace=False,
            )
            # flip negative_sample[flip_sites[0]] and negative_sample[flip_sites[1]]
            negative_sample[flip_sites[0]], negative_sample[flip_sites[1]] = (
                negative_sample[flip_sites[1]],
                negative_sample[flip_sites[0]],
            )
            assert len(negative_sample) == len(positive_sample)
            assert (
                negative_sample[: int(len(positive_sample) / 2)]
                == positive_sample[: int(len(positive_sample) / 2)]
            )
            assert set(negative_sample[int(len(positive_sample) / 2) :]) == set(
                positive_sample[int(len(positive_sample) / 2) :]
            )

        negative_samples.append(negative_sample)

        if len(negative_samples) % 1000 == 0:
            print(f"generated {len(negative_samples)} negative samples...")
    assert len(negative_samples) == len(
        positive_samples
    ), f"{len(negative_samples)} != len(positive_samples) (={len(positive_samples)})"
    return negative_samples


def generate_dependency_samples(
    data_size: int,
    num_of_terms: int,
    num_of_nonterms: int,
    min_dep: int,
    max_dep: int,
    dependency_type: str,
    erase_duplicated_data: bool = False,
    dist: str = "uniform",
) -> Tuple[List[List[str]], List[List[str]]]:
    assert data_size % 2 == 0, "data_size must be even."
    positive = generate_dependency_positive_samples(
        int(data_size / 2),
        num_of_terms,
        num_of_nonterms,
        min_dep,
        max_dep,
        dependency_type,
        erase_duplicated_data=erase_duplicated_data,
        dist=dist,
    )
    negative = generate_dependency_negative_samples(
        positive,
    )
    return positive, negative
