#!/usr/bin/env python
"""Top-level evaluation entry point (thin wrapper around src/evaluate.py).

Reports precision, recall, F1 and FAR on three test slices:
clean / noisy / adversarial. See `python evaluate.py --help`.
"""

from src.evaluate import main

if __name__ == "__main__":
    main()
