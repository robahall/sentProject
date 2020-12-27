
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import GPUStatsMonitor

from src.models.lstm import LSTM
from src.data.dataloader import IMDBDataModule




def main(hparams):
    exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = TensorBoardLogger(save_dir='logs/',
                                  version=f'v_{exp_time}')

    gpu_stats = GPUStatsMonitor()


    data = IMDBDataModule(hparams=hparams)
    data.prepare_data()
    data.setup()

    model = LSTM(input_dim=data.dims,
                 hparams=hparams,
                 pad_idx=data.pad_idx
                 )

    model.embedding.weight.data.copy_(data.text.vocab.vectors) # Load pre-trained vectors
    model.embedding.weight.data[data.unk_idx] = torch.zeros(hparams.embedding_dim)
    model.embedding.weight.data[data.pad_idx] = torch.zeros(hparams.embedding_dim)

    trainer = Trainer(logger=tb_logger,
                      gpus=hparams.gpus,
                      max_epochs=hparams.max_epochs,
                      callbacks=[gpu_stats])

    trainer.fit(model, data)

    trainer.test(datamodule=data)

    torch.save(trainer.model.state_dict(), f'models/model_{exp_time}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgraded Sentiment Analysis",
                                     add_help=True)

    parser.add_argument(
        "--min_epochs",
        default=5,
        type=int,
        help="Min number of epochs to try."
    )

    parser.add_argument(
        "--max_epochs",
        default=5,
        type=int,
        help="Max number of epochs to try."
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs to use (Currently 0 or 1).")

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size to be used."
    )
    parser.add_argument(
        "--deterministic",
        action='store_false',
        help='Utilize only deterministic algorithms (repeatable results). '
    )
    parser.add_argument(
        "--include_lengths",
        action='store_false',
        help="Forces a NN to process only the non-padded elements in sequence.",
    )

    parser = LSTM.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)







