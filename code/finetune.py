from glob import glob
import json
import optuna
import os
import pandas as pd
from PIL import Image
import shutil
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


from utils import *
from conch.open_clip_custom import get_tokenizer, tokenize
from evaluation import zero_shot_eval


class ClipLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        labels = torch.arange(
            logits_per_image.shape[0], device=image_features.device, dtype=torch.long)

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss


class FineTuneModel(pl.LightningModule):
    def __init__(self, model, T_max, lr, wd):
        super().__init__()
        self.model = model
        self.T_max = T_max
        self.lr = lr
        self.wd = wd
        self.loss_fn = ClipLoss()

    def forward(self, image, text):
        # Encode both images and text
        image_embeddings = self.model.encode_image(image, normalize=True)
        text_embeddings = self.model.encode_text(text, normalize=True)
        return image_embeddings, text_embeddings

    def training_step(self, batch, batch_idx):
        images, captions = batch
        image_embeddings, text_embeddings = self(images, captions)
        loss = self.loss_fn(
            image_embeddings, text_embeddings, self.model.logit_scale)
        self.log("train_loss", loss, on_epoch=True)

        # Log current learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.T_max)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def finetune(vl_model, task, image_caption_source, 
             lr, wd, T_max=50, max_epochs=None,
             parameter_tuning=False, training_images=None, image_dir=None,
             eval_only=False, eval_ckp=None):
    
    model, preprocess, tokenizer = get_vl_model(vl_model, ckp=eval_ckp)
    finetune_model = FineTuneModel(model=model, T_max=T_max, lr=lr, wd=wd)

    if eval_only:
        results = zero_shot_eval(vl_model, task, model, preprocess, tokenizer)
        return results

    retrieve_csv = f"CONCH/results/{task}_{image_caption_source}.csv"
    ds = ImageCaptionDataset(vl_model=vl_model,
                             retrieve_csv=retrieve_csv,
                             tokenizer=tokenizer,
                             transform=preprocess,
                             training_images=training_images,
                             image_dir=image_dir)
    dl = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)

    # Trainer
    if not parameter_tuning:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer(callbacks=[EarlyStopping(monitor='train_loss', mode='min', patience=5),
                                        lr_monitor],
                             max_epochs=max_epochs,
                             accelerator='gpu',
                             enable_model_summary=1,
                             logger=TensorBoardLogger("logs/", name=f"{vl_model}_finetune", version=f"{task}_{image_caption_source}"))

    else:
        trainer = pl.Trainer(max_epochs=5, accelerator='gpu',
                             enable_checkpointing=False)

    # Train the model
    trainer.fit(finetune_model, train_dataloaders=dl)

    if not parameter_tuning:
        results = zero_shot_eval(vl_model, task, model, preprocess, tokenizer)
        return results
    else:
        return trainer


def finetune_parameter_tuning(vl_model, task, image_caption_source, training_images, image_dir=None):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        trainer = finetune(vl_model, task, image_caption_source, lr, wd,
                           parameter_tuning=True,
                           training_images=training_images,
                           image_dir=image_dir)
        return trainer.callback_metrics["train_loss"].item()

    study = optuna.create_study(direction="minimize")  # Minimize loss
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters:", study.best_params)
    return study.best_params
