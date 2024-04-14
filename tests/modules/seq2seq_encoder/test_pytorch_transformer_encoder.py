import torch

from chomsky_neural.modules.seq2seq_encoder import PytorchTransformerEncoder


def test_pytorch_transformer_encoder():
    encoder = PytorchTransformerEncoder(
        input_dim=5,
        num_layers=1,
        feedforward_hidden_dim=10,
        num_attention_heads=1,
        positional_encoding="sinusoidal",
        positional_embedding_size=5,
    )
    encoder.eval()
    inputs = torch.rand(3, 4, 5)  # (batch_size, sequence_length, input_dim)
    mask = torch.ones(3, 4).bool()

    output = encoder(inputs, mask)
    assert output.size() == (3, 4, 5)



    # check if causal mask is generated correctly
    # prepare inputs for auto-regressive test. create inputs but with different content where seq_len > 1
    # check if the output is the same as the input  for the first token
    encoder._auto_regressive = True
    output = encoder(inputs, mask)
    assert output.size() == (3, 4, 5)
    inputs_with_different_content = inputs.clone()
    inputs_with_different_content[:, 1:, :] = torch.rand(3, 3, 5)
    output_with_different_content = encoder(inputs_with_different_content, mask)
    assert torch.allclose(output[:, 0, :], output_with_different_content[:, 0, :])
