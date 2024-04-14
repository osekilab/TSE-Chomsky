import torch
from chomsky_neural.data.instance import PairBatch
from chomsky_neural.models.seq2seq_with_lm_head import Seq2SeqWithLMHead
from chomsky_neural.modules.seq2seq_encoder.seq2seq_encoder import Seq2SeqEncoder
from chomsky_neural.modules.token_embedder.token_embedder import TokenEmbedder
from chomsky_neural.torch.optimizer import Optimizer
from tango.common import Lazy


def test_seq2seq_with_lm_head():
    token_embedder = TokenEmbedder.from_params(
        params_={
            "type": "embedding",
            "embedding_dim": 8,
            "num_embeddings": 10,
        }
    )
    seq2seq_encoder = Seq2SeqEncoder.from_params(
        params_={
            "type": "lstm",
            "input_size": 8,
            "hidden_size": 8,
            "num_layers": 1,
            "bidirectional": False,
            "dropout": 0.0,
        }
    )

    model = Seq2SeqWithLMHead(
        token_embedder=token_embedder,
        seq2seq_encoder=seq2seq_encoder,
        vocab_size=10,
        optimizer=Lazy(Optimizer, params={"type": "torch::Adam", "lr": 0.001}),
        scheduler=None,
    )
    inputs = torch.randint(
        low=0,
        high=10,
        size=(3, 5),
    )
    mask = torch.ones(3, 5).bool()
    output = model(inputs, mask)
    assert output.size() == (3, 5, 10)

    # calcualte likelihood
    likelihood = model.calculate_likelihood(inputs, mask)
    assert len(likelihood) == 3
    assert len(likelihood[0]) == 5  # sequence_length - 1

    # calculate likelihood for pairs
    pair_batch = PairBatch(
        inputs_pos=inputs,
        inputs_neg=inputs,
        mask_pos=mask,
        mask_neg=mask,
    )
    outputs = model.calculate_likelihood_for_pair_batch(pair_batch)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output["good_likelihood"]) == 5
        assert len(output["bad_likelihood"]) == 5
