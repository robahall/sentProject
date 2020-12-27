
import argparse
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import GPUStatsMonitor

from src.models.rnn import RNN
from src.data.dataloader import IMDBDataModule


def main(hparams):

    exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = TensorBoardLogger(save_dir='logs/',
                                  version=f'v_{exp_time}')

    gpu_stats = GPUStatsMonitor()


    data = IMDBDataModule(hparams=hparams)
    data.prepare_data()
    data.setup()

    model = RNN(input_dim=data.dims,
                hparams=hparams)

    trainer = Trainer(logger=tb_logger,
                      gpus=hparams.gpus,
                      max_epochs=hparams.max_epochs,
                      callbacks=[gpu_stats])

    trainer.fit(model, data)

    trainer.test(datamodule=data)

    torch.save(trainer.model.state_dict(), f'models/model_{exp_time}.pth')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simple Sentiment Analysis",
                                     add_help=True)

    parser.add_argument(
        "--min_epochs",
        default=5,
        type=int,
        help= "Min number of epochs to try."
        )

    parser.add_argument(
        "--max_epochs",
        default=5,
        type=int,
        help= "Max number of epochs to try."
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

    parser = RNN.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)







