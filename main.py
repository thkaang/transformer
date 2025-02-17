import logging
import os
import torch
from torch import nn, optim
from train import build_default_model
from data import Multi30k
from config import *

DATASET = Multi30k()


def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0

    for idx, (src, tgt) in enumerate(data_loader):
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        tgt_x = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        optimizer.zero_grad()

        output, _ = model(src, tgt_x)

        y_hat = output.contiguous().view(-1, output.shape[-1])  # check shape
        y_gt = tgt_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
    num_samples = idx + 1

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"{epoch:04d}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, checkpoint_file)

    return epoch_loss / num_samples


def main():
    model = build_default_model(len(DATASET.vocab_src), len(DATASET.vocab_tgt), device=DEVICE, dr_rate=DROPOUT_RATE)

    def initialize_weights(model):
        if hasattr(model, 'weight') and model.weight.dim() > 1:
            nn.init.kaiming_uniform_(model.weight.data)

    model.apply(initialize_weights)

    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    criterion = nn.CrossEntropyLoss(ignore_index=DATASET.pad_idx)

    train_iter, valid_iter, test_iter = DATASET.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    for epoch in range(N_EPOCH):
        logging.info(f"*****epoch: {epoch:02}*****")
        train_loss = train(model, train_iter, optimizer, criterion, epoch, CHECKPOINT_DIR)
        logging.info(f"train_loss: {train_loss:.5f}")

if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    main()
