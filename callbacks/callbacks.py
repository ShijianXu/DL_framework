import os
from . import Callback

class CheckpointResumeCallback(Callback):
    def __init__(self, resume):
        self.resume = resume

    def on_train_begin(self, trainer):
        ckpt_path = os.path.join(trainer.log_dir, 'checkpoints', 'checkpoint_latest.pth')
        if self.resume and os.path.isfile(ckpt_path):
            print("=> Resuming ckpt ...")
            trainer.resume_ckpt(ckpt_path)
        else:
            print("=> No ckpt in log dir. Starting training from scratch.")


class CheckpointSaveCallback(Callback):
    def __init__(self, every_n_epochs=None):
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, epoch):
        if self.every_n_epochs is not None and (epoch + 1) % self.every_n_epochs == 0:
            trainer.save_checkpoint(epoch)
            print("=> Checkpoint saved.")