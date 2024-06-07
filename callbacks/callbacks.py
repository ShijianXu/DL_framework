import os
from . import Callback
import time

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


class TrainingTimerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = 0

    def on_train_begin(self, trainer):
        # This method is called when the training starts.
        self.start_time = time.time()
        print("=> Training started...")

    def on_train_end(self, trainer):
        # This method is called when the training ends.
        time_elapsed = time.time() - self.start_time
        print('Total time elapsed: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))