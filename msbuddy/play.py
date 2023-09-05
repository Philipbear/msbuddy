import torch
import torch.nn as nn
import torch.nn.functional as F


class MS2SpectrumRanker(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_encoder_layers=3):
        super(MS2SpectrumRanker, self).__init__()
        self.linear_projection = nn.Linear(10, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        x = self.linear_projection(x)  # Shape: (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch_size, d_model)

        # Apply the mask if provided
        if mask is not None:
            x = self.encoder(x, src_key_padding_mask=mask)  # Corrected here
        else:
            x = self.encoder(x)

        x = torch.mean(x, dim=0)  # Aggregation (Mean): Shape (batch_size, d_model)
        x = self.linear(x)  # Scoring layer: Shape (batch_size, 1)

        return torch.sigmoid(x)  # Output between 0 and 1


# test
if __name__ == '__main__':
    # Initialize the model
    model = MS2SpectrumRanker()

    # Dummy input for testing: Batch of 5 spectra,
    # each with variable-length fragments (<= 50), each fragment a 10D vector
    dummy_input1 = torch.rand(30, 10)
    dummy_input2 = torch.rand(45, 10)
    dummy_input3 = torch.rand(50, 10)
    dummy_input4 = torch.rand(20, 10)
    dummy_input5 = torch.rand(40, 10)

    # Pad to the same sequence length (50)
    padded_inputs = [F.pad(input, (0, 0, 0, 50 - input.shape[0])) for input in
                     [dummy_input1, dummy_input2, dummy_input3, dummy_input4, dummy_input5]]

    # Create masks for padding
    masks = [[False] * input.shape[0] + [True] * (50 - input.shape[0]) for input in
             [dummy_input1, dummy_input2, dummy_input3, dummy_input4, dummy_input5]]

    # Convert to tensor
    padded_inputs = torch.stack(padded_inputs)
    masks = torch.tensor(masks)

    # Forward pass
    output = model(padded_inputs, mask=masks)
    print("Output Shape:", output.shape)
    print("Output Values:", output)

