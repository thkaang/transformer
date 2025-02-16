import logging
import torch
from train import build_default_model
from data import Multi30k
from config import *

DATASET = Multi30k()


def main():
    model = build_default_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)


if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    main()
