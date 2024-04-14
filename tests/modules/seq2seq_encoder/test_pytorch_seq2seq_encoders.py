import torch
from chomsky_neural.modules.seq2seq_encoder.pytorch_seq2seq_wrapper import (
    GruSeq2SeqEncoder,
    LstmSeq2SeqEncoder,
    RnnSeq2SeqEncoder,
)


def test_gru_seq2seq_encoder():
    encoder = GruSeq2SeqEncoder(
        input_size=4,
        hidden_size=8,
        num_layers=1,
    )
    inputs = torch.rand(2, 3, 4)  # (batch_size, sequence_length, input_size)
    mask = torch.ones(2, 3).bool()
    output = encoder(inputs, mask)
    assert output.size() == (2, 3, 8)


def test_lstm_seq2seq_encoder():
    encoder = LstmSeq2SeqEncoder(
        input_size=4,
        hidden_size=8,
        num_layers=1,
    )
    inputs = torch.rand(2, 3, 4)  # (batch_size, sequence_length, input_size)
    mask = torch.ones(2, 3).bool()
    output = encoder(inputs, mask)
    assert output.size() == (2, 3, 8)


def test_rnn_seq2seq_encoder():
    encoder = RnnSeq2SeqEncoder(
        input_size=4,
        hidden_size=8,
        num_layers=1,
    )
    inputs = torch.rand(2, 3, 4)  # (batch_size, sequence_length, input_size)
    mask = torch.ones(2, 3).bool()
    output = encoder(inputs, mask)
    assert output.size() == (2, 3, 8)
