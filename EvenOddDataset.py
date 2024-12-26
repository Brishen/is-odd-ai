import torch
from torch.utils.data import Dataset


class EvenOddDataset(Dataset):
    """
    A PyTorch Dataset that returns a sequence of digit tokens and a label (0=even, 1=odd).
    Includes support for negative numbers by prepending a minus token.
    """

    def __init__(self, numbers, max_length=7):
        """
        numbers: list of integers (positive, negative, or zero)
        max_length: pad/truncate sequences to this length
        """
        super().__init__()
        self.max_length = max_length

        self.samples = []
        for num in numbers:
            abs_num = abs(num)
            label = 0 if (abs_num % 2) == 0 else 1  # even=0, odd=1

            # Convert the absolute value to a list of digit tokens
            digit_str = str(abs_num)
            digit_tokens = [int(d) for d in digit_str] if digit_str != '0' else [0]

            # If num is negative, prepend a special MINUS token (index=11)
            # We'll define: digits=0..9, PAD=10, MINUS=11
            if num < 0:
                digit_tokens = [11] + digit_tokens  # Prepend the MINUS token

            self.samples.append((digit_tokens, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        digits, label = self.samples[idx]

        # Pad or truncate to max_length
        # We'll define the PAD token as index=10
        if len(digits) < self.max_length:
            digits += [10] * (self.max_length - len(digits))  # pad
        else:
            digits = digits[:self.max_length]  # truncate

        return torch.tensor(digits, dtype=torch.long), torch.tensor(label, dtype=torch.long)
