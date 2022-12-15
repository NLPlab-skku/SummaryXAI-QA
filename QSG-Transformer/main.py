# -*- coding: utf-8 -*-

import hydra
from omegaconf import DictConfig

from src.datamodule import *
from src.learner import Learner
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(config_path = '.', config_name = 'config')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    datamodule = DataModule(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    checkpoint_callback = ModelCheckpoint(
        dirpath = cfg.output_dir,
        monitor = 'val_rouge2',
        save_top_k = 1,
        mode = 'max'
    )
    early_stopping = EarlyStopping(
        monitor = 'val_rouge2',
        patience = cfg.patience,
        mode = 'max'
    )
    logger = TensorBoardLogger(
        save_dir = cfg.output_dir,
        name = 'tb_logs'
    )

    learner = Learner(cfg.learner)

    trainer = Trainer(
        accelerator = 'gpu',
        devices = [cfg.devices],
        logger = logger,
        max_steps = cfg.num_training_steps,
        max_epochs = -1,
        check_val_every_n_epoch = cfg.check_val_every_n_epoch,
        callbacks = [checkpoint_callback, early_stopping]
    )

    trainer.fit(
        learner,
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader
    )

    trainer.test(
        model = learner,
        dataloaders = test_dataloader,
        ckpt_path = 'best'
    )

if __name__ == '__main__':
    main()