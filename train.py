import csv
import os
from collections import OrderedDict

from tqdm import tqdm

from dataset.cad_dataset import get_dataloader
from config import ConfigAE
from utils import cycle, ensure_dir
from trainer import TrainerAE


def append_csv_row(csv_path, row):
    ensure_dir(os.path.dirname(csv_path))
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    cfg = ConfigAE('train')

    tr_agent = TrainerAE(cfg)

    if cfg.cont:
        tr_agent.load_ckpt(cfg.ckpt)

    train_loader = get_dataloader('train', cfg)
    val_loader = get_dataloader('validation', cfg)
    val_loader_all = get_dataloader('validation', cfg)
    val_loader = cycle(val_loader)

    clock = tr_agent.clock
    metrics_csv = os.path.join(cfg.exp_dir, 'artifacts', 'train_metrics.csv')

    for e in range(clock.epoch, cfg.nr_epochs + 1):
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            outputs, losses = tr_agent.train_func(data)
            train_loss_cmd = losses['loss_cmd'].item()
            train_loss_args = losses['loss_args'].item()

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            if clock.step % cfg.val_frequency == 0:
                data = next(val_loader)
                tr_agent.val_func(data)

            clock.tick()

            append_csv_row(
                metrics_csv,
                {
                    'epoch': e,
                    'step': clock.step,
                    'loss': train_loss_cmd + train_loss_args,
                    'loss_cmd': train_loss_cmd,
                    'loss_args': train_loss_args,
                },
            )

            tr_agent.update_learning_rate()

        if clock.epoch % 5 == 0:
            tr_agent.evaluate(val_loader_all)

        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()

        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
