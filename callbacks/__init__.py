class Callback(object):
    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, epoch):
        pass

    def on_train_end(self, trainer):
        pass

    def on_batch_end(self, trainer, batch_idx):
        pass


# add all the callbacks here
from .callbacks import CheckpointResumeCallback