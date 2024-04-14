import torch

from chomsky_neural.modules.seq2seq_encoder.stack_rnn_encoder import \
    StackRNNSeq2SeqEncoder


def test_stack_rnn_seq2seq_encoder():
    encoder = StackRNNSeq2SeqEncoder(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        batch_first=True,
        stack_cell_size=4,
        stack_size=8,
        n_stacks=1,
    )
    inputs = torch.rand(2, 3, 4)  # (batch_size, sequence_length, input_size)
    mask = torch.ones(2, 3).bool()
    output = encoder(inputs, mask)
    assert output.size() == (2, 3, 8)
