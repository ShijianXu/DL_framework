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

