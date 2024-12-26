# Oddness Prediction AI

This repository contains a machine learning model to classify integers as either odd or even. The model is implemented using PyTorch and a custom Transformer architecture.

## Features

- Uses a Transformer model for digit sequence classification.
- Handles both positive and negative integers, as well as zero.
- Provides a CLI interface to test numbers for oddness.

## Repository Structure

- `EvenOddDataset.py`: Defines the custom dataset for handling sequences of digits.
- `DigitTransformer.py`: Implements the Transformer-based model for sequence classification.
- `train.py`: Script for training the model.
- `cli.py`: Command-line interface for running inference on numbers.

## Getting Started

### Prerequisites

- Python 3.8+

### Installation

To install the project using the provided `pyproject.toml` file, run the following command:

```bash
pip install .
```

Alternatively, if you are developing the project, install it in editable mode:

```bash
pip install -e .
```

### Training the Model

1. Generate a large dataset of random integers.
2. Train the model using `train.py`:
   ```bash
   python train.py
   ```
   This script will save the trained model as `my_transformer_model.pt`.

### Running Inference

Use the CLI to test numbers for oddness:
```bash
python cli.py [NUMBER]
```

For example:
```bash
python cli.py 42
```

Output:
```
Number: 42 => Predicted: Even
```

### Options

- `--model-path`: Path to the trained model file (default: `my_transformer_model.pt`).

## Example Workflow

1. Train the model using `train.py`.
2. Use `cli.py` to test the trained model on integers.

## Model Details

- The model takes a sequence of digit tokens as input.
- Uses a vocabulary size of 12 (digits 0-9, PAD=10, MINUS=11).
- Pads or truncates sequences to a fixed length of 7.

## Contributing

Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
