import argparse
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pytorch_lightning as pl
import torch
from chomsky_neural.data.source import ChomskyDataSource, ChomskyPairDataSource
from chomsky_neural.data.vocabulary import PAD_TOKEN
from chomsky_neural.env import DATA_DIR, OUTPUT_DIR, PACKAGE_DIR
from chomsky_neural.file_utils import load_json, load_jsonlines, save_as_jsonlines
from chomsky_neural.models.seq2seq_with_lm_head import Seq2SeqWithLMHead
from chomsky_neural.modules.datamodule.datamodule import ChomskyDatamodule
from chomsky_neural.modules.datamodule.pair_datamodule import ChomskyPairDatamodule
from chomsky_neural.modules.seq2seq_encoder import Seq2SeqEncoder
from chomsky_neural.modules.token_embedder.token_embedder import TokenEmbedder
from lightning_lite.utilities.seed import seed_everything
from tango.common import Lazy
from tango.integrations.torch import Optimizer
from tqdm import tqdm


def append_likelihood_to_data(
    data_path: Union[Path, str], output_path: Union[Path, str], predictions: list
) -> List[Dict]:
    data = load_jsonlines(data_path)
    for i, datum in enumerate(data):
        datum["good_likelihood"] = predictions[i]["good_likelihood"]
        datum["bad_likelihood"] = predictions[i]["bad_likelihood"]
        datum["good_loglik"] = np.where(
            np.isnan(np.log(predictions[i]["good_likelihood"])),
            -np.inf,
            np.log(predictions[i]["good_likelihood"]),
        ).tolist()
        datum["bad_loglik"] = np.where(
            np.isnan(np.log(predictions[i]["bad_likelihood"])),
            -np.inf,
            np.log(predictions[i]["bad_likelihood"]),
        ).tolist()
        datum["good_avg_likelihood"] = np.mean(datum["good_likelihood"][1:])
        datum["bad_avg_likelihood"] = np.mean(datum["bad_likelihood"][1:])
        datum["good_avg_loglik"] = np.mean(datum["good_loglik"][1:])
        datum["bad_avg_loglik"] = np.mean(datum["bad_loglik"][1:])
    save_as_jsonlines(data, output_path)
    return data


def model_type2name(model_config: Dict[str, Any]) -> str:
    model_type = model_config["type"]
    if model_type == "lstm":
        if model_config["bidirectional"]:
            model_name = "BiLSTM"
        else:
            model_name = "LSTM"
        return "{}_{}layers_{}input_{}hidden".format(
            model_name,
            model_config["num_layers"],
            model_config["input_size"],
            model_config["hidden_size"],
        )
    elif model_type == "pytorch_transformer_encoder":
        if model_config["auto_regressive"]:
            model_name = "TransformerDecoder"
        else:
            model_name = "TransformerEncoder"
        return "{}_{}layers_{}dim_{}heads_{}FF".format(
            model_name,
            model_config["num_layers"],
            model_config["input_dim"],
            model_config["num_attention_heads"],
            model_config["feedforward_hidden_dim"],
        )
    elif model_type == "bert_encoder":
        model_name = "BERT"
        return "{}_{}layers_{}dim_{}heads_{}FF_{}-pooling".format(
            model_name,
            model_config["num_layers"],
            model_config["input_dim"],
            model_config["num_attention_heads"],
            model_config["feedforward_hidden_dim"],
            model_config["pooling_method"],
        )
    elif model_type == "stack_rnn":
        model_name = "StackRNN"
        return "{}_{}layers_{}input_{}hidden_{}stacksize".format(
            model_name,
            model_config["num_layers"],
            model_config["input_size"],
            model_config["hidden_size"],
            model_config["stack_size"],
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=lambda p: Path(p).absolute(),
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="chomsky_neural",
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="AnBn",
        help="Task for training.",
    )
    parser.add_argument(
        "--num_terms",
        type=int,
        default=2,
        help="Number of terms for training.",
    )
    parser.add_argument(
        "--num_of_nonterms",
        type=int,
        default=5,
        help="Number of nonterminals.",
    )
    parser.add_argument(
        "--min_iter",
        type=int,
        default=2,
        help="Minimum number of iterations in train/dev data.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=15,
        help="Maximum number of iterations in train/dev data.",
    )
    parser.add_argument(
        "--max_iter_test",
        type=int,
        default=50,
        help="Maximum number of iterations in test data.",
    )
    parser.add_argument(
        "--in_dist_data_size",
        type=int,
        default=50000,
        help="Size of in-dist data.",
    )
    parser.add_argument(
        "--erase_duplicated_data",
        action="store_true",
        help="Erase duplicated data.",
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="uniform",
        help="Distribution of the terminals.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=15,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Accelerator for training.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=-1,
        help="Number of GPUs for training.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Strategy for training.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.1,
        help="Validation check interval.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=10240, help="Batch size for evaluation."
    )
    args = parser.parse_args()
    config = load_json(args.config)
    seed_everything(args.seed)

    if args.task not in ["AnBn", "AnB"]:
        dataname = f"{args.task}_{args.num_terms}terms_{args.num_of_nonterms}nonterms_{args.min_iter}_{args.max_iter}_{args.max_iter_test}_{'erase-dup' if args.erase_duplicated_data else 'keep-dup'}_{args.dist}"
    else:
        dataname = f"{args.task}_{args.num_terms}terms_{args.min_iter}_{args.max_iter}_{args.max_iter_test}_{'erase-dup' if args.erase_duplicated_data else 'keep-dup'}_{args.dist}"
    train_datasource = ChomskyDataSource(
        dataname=dataname, subset="train", correct_only=True
    )
    validation_datasource = ChomskyDataSource(
        dataname=dataname, subset="validation", correct_only=True
    )
    test_datasource = ChomskyDataSource(
        dataname=dataname, subset="test", correct_only=True
    )
    test_pair_datasource = ChomskyPairDataSource(dataname=dataname, subset="test_pair")
    ood_pair_datasource = ChomskyPairDataSource(
        dataname=dataname, subset="out_of_dist_test_pair"
    )
    datamodule = ChomskyDatamodule(
        train_datasource=train_datasource,
        val_datasource=validation_datasource,
        test_datasource=test_datasource,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.build_vocab()
    model_config = config["model"]
    model_name = model_type2name(model_config)
    optimizer = config["optimizer"]

    seq2seq_encoder = Seq2SeqEncoder.from_params(params_=model_config.copy())
    token_embedder = TokenEmbedder.from_params(
        params_={
            "type": "embedding",
            "embedding_dim": seq2seq_encoder.get_input_dim(),
            "num_embeddings": datamodule.vocab.size("tokens"),
            "padding_idx": datamodule.vocab.get(PAD_TOKEN),
        }
    )
    optimizer = Lazy(Optimizer, params={"type": optimizer, "lr": args.learning_rate})
    model = Seq2SeqWithLMHead(
        token_embedder=token_embedder,
        seq2seq_encoder=seq2seq_encoder,
        vocab_size=datamodule.vocab.size("tokens"),
        optimizer=optimizer,
    )
    output_dir = (
        OUTPUT_DIR
        / f"{args.task}_{'erase-dup' if args.erase_duplicated_data else 'keep-dup'}_{args.dist}_{model_name}_{args.num_of_nonterms}_{args.num_terms}terms_{args.learning_rate}"
        / f"seed{args.seed}"
    )
    tensorboard_logger = pl.loggers.TensorBoardLogger(
        save_dir=output_dir / "logs",
    )
    mlflow_logger = pl.loggers.MLFlowLogger(
        experiment_name=args.exp_name,
        run_name=f"{args.task}_{'erase-dup' if args.erase_duplicated_data else 'keep-dup'}_{args.dist}_{model_name}_{args.num_of_nonterms}_{args.num_terms}terms_{args.learning_rate}_seed{args.seed}",
        save_dir=str(PACKAGE_DIR / "mlruns"),
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        logger=[tensorboard_logger, mlflow_logger],
        callbacks=[checkpoint_callback],
        deterministic=True,
        max_epochs=args.num_epochs,
        val_check_interval=args.val_check_interval,
    )

    trainer.fit(model, datamodule=datamodule)
    best_checkpoint_path = checkpoint_callback.best_model_path
    best_checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(best_checkpoint["state_dict"])

    # pred/evaluate
    pair_datamodule = ChomskyPairDatamodule(vocab=datamodule.vocab)
    test_pair_dataloader = pair_datamodule.build_dataloader_from_source(
        datasource=test_pair_datasource,
        batch_size=args.eval_batch_size,
    )
    ood_pair_dataloader = pair_datamodule.build_dataloader_from_source(
        datasource=ood_pair_datasource,
        batch_size=args.eval_batch_size,
    )
    test_predictions = []
    ood_predictions = []
    for pair_batch in tqdm(
        test_pair_dataloader, desc="test", total=len(test_pair_dataloader)
    ):
        test_predictions.extend(model.calculate_likelihood_for_pair_batch(pair_batch))
    for pair_batch in tqdm(
        ood_pair_dataloader, desc="ood", total=len(ood_pair_dataloader)
    ):
        ood_predictions.extend(model.calculate_likelihood_for_pair_batch(pair_batch))

    # save prediction
    test_predictions = append_likelihood_to_data(
        DATA_DIR / dataname / "test_pair.jsonl",
        output_dir / "test_pred.jsonl",
        test_predictions,
    )
    ood_predictions = append_likelihood_to_data(
        DATA_DIR / dataname / "out_of_dist_test_pair.jsonl",
        output_dir / "ood_pred.jsonl",
        ood_predictions,
    )
    combined_prediction = test_predictions + ood_predictions
    save_as_jsonlines(combined_prediction, output_dir / "combined_pred.jsonl")
