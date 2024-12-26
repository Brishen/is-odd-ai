import click
import torch
from pathlib import Path
from DigitTransformer import DigitTransformer

def load_model(model_path, device):
    """Load the trained model."""
    model = DigitTransformer(
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
        max_length=7, vocab_size=12, num_classes=2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def predict_oddness(model, number, device):
    """Run inference to determine if a number is odd or even."""
    abs_num = abs(number)

    # Tokenize
    digit_str = str(abs_num)
    digits = [int(d) for d in digit_str] if digit_str != "0" else [0]
    if number < 0:
        digits = [11] + digits  # MINUS token

    # Pad/truncate to max_length=7
    if len(digits) < 7:
        digits += [10] * (7 - len(digits))  # PAD
    else:
        digits = digits[:7]

    x_tensor = torch.tensor([digits], dtype=torch.long).to(device)
    logits = model(x_tensor)
    pred_idx = torch.argmax(logits, dim=-1).item()

    return "Odd" if pred_idx == 1 else "Even"

@click.command()
@click.argument('number', type=int)
@click.option('--model-path', default='my_transformer_model.pt', help='Path to the trained model file.')
def cli(number, model_path):
    """CLI to check if a NUMBER is odd or even using the trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(model_path).exists():
        click.echo(f"Error: Model file '{model_path}' not found.")
        return

    model = load_model(model_path, device)
    result = predict_oddness(model, number, device)
    click.echo(f"Number: {number} => Predicted: {result}")

if __name__ == '__main__':
    cli()
