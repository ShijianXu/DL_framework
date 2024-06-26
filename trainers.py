import os
import torch
from tqdm import tqdm
import utils
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import torchvision.utils as vutils

class Trainer(object):
    def __init__(self,
        config,
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        epochs,
        scheduler=None,
        sample_valid=False,
        sample_valid_freq=5,
        print_freq=400,
        log_dir='./logs',
        resume=True,
        resume_optimizer=True,
        log_tool='tensorboard',
        callbacks=None,
        eval_metric=None
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        if self.val_dataloader is None:
            print("=> No validation dataloader provided. Skip validation at the end of each epoch.")
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.start_epoch = 0
        self.total_steps = 0

        self.print_freq = print_freq
        self.log_dir = log_dir
        self.resume = resume
        self.resume_optimizer = resume_optimizer

        self.sample_valid = sample_valid            # indicate whether to generate images for validation
        self.sample_valid_freq = sample_valid_freq  # epoch frequency to generate images for validation

        self.callbacks = callbacks
        assert self.callbacks is not None, "Basic callbacks must be provided."

        self.eval_metric = eval_metric

        # init logger
        self.log_tool = log_tool
        if self.log_tool == 'tensorboard':
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "log"))
        elif self.log_tool == 'wandb':
            import wandb
            wandb.init(project='random_project')
            self.writer = wandb
        else:
            raise ValueError("Log tool not supported.")

        # check device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # to device
        self.model.to(self.device)
        self.model.everything_to(self.device)       # move some internal variables to device
        self.criterion.to(self.device)

    def train(self):
        # from pudb import set_trace; set_trace()

        for callback in self.callbacks:
            callback.on_train_begin(self)

        try:
            self.losses_m = utils.AverageMeter()
            for epoch in range(self.start_epoch, self.epochs):
                for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                    self.process_batch(batch)

                    if batch_idx % self.print_freq == 0:
                        print("Epoch: {}, batch: {}, train loss: {:.5f}".format(epoch, batch_idx, self.losses_m.avg))
            
                if self.val_dataloader is not None:
                    valid_loss = self.validate(epoch)
                    print("Epoch: {}, valid loss: {:.5f}".format(epoch, valid_loss))

                if self.sample_valid and epoch % self.sample_valid_freq == 0:
                    sample_tensor = self.model.generate(64, self.device)
                    self.log_images("Valid/Sample", sample_tensor, epoch)

                if self.scheduler is not None:
                    if hasattr(self.config, 'scheduler_name') and self.config.scheduler_name == 'ReduceLROnPlateau':
                        self.scheduler.step(valid_loss)
                    else:
                        self.scheduler.step()

                lr = self.optimizer.param_groups[0]["lr"]
                self.log_scalar("Train/Learning rate", lr, epoch)

                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch)

                self.losses_m.reset()

            self.save_checkpoint(epoch)         # make sure the latest checkpoint is saved even if no callback is used
            print("=> Training Finished. Final checkpoint saved.")
            
            if self.log_tool == 'tensorboard':
                self.writer.close()
            
        except KeyboardInterrupt:
            self.save_checkpoint(epoch)
            print("=> Traing interrupted. Checkpoint saved.")

        for callback in self.callbacks:
            callback.on_train_end(self)

    def process_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        # The returned loss is a dict
        losses = self.model.process_batch(batch, self.criterion, self.device)

        # log loss
        for key, val in losses.items():
            self.log_scalar(f"Train/{key}", val.item(), self.total_steps)

        # update loss AverageMeter
        self.losses_m.update(losses['loss'].item(), batch[0].size(0))
        
        losses['loss'].backward()
        self.optimizer.step()

        # EMA update for NCSNv2
        if hasattr(self.model, 'ema_helper'):
            self.model.post_update()

        self.total_steps += 1

    def validate(self, epoch):
        self.model.eval()
        self.model.reset_metric()
        losses_v = utils.AverageMeter()
        print("=> Validating ...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader)):
                source, target = batch
                source = source.to(self.device)
                target = target.to(self.device)

                output = self.model(source)
                losses = self.model.compute_loss(output, target, self.criterion)
                losses_v.update(losses['loss'].item(), source.size(0))

                # compute metrics
                # source is needed for some generative models
                self.model.compute_metric(source, output, target, self.eval_metric)

        # check if the metric is the best
        # update the best metric inside the model and save the checkpoint
        if self.model.is_best_metric():
            print("=> Best metric achieved. Saving checkpoint ...")
            self.save_checkpoint(epoch, filename='checkpoint_best.pth')

        self.log_scalar("Valid/Loss", losses_v.avg, epoch)
        self.log_scalar("Valid/Metric", self.model.get_metric_value(), epoch)
        self.model.display_metric_value()

        if hasattr(self.config, 'valid_sample'):
            if self.config.valid_sample:
                print("Validation sampling ...")
                self._on_valid_end(epoch)

        return losses_v.avg

    def _on_valid_end(self, epoch):
        test_input, _ = next(iter(self.val_dataloader))
        test_input = test_input.to(self.device)
        
        recons = self.model.generate(test_input)
        recons_dir = os.path.join(self.log_dir, "Recons")
        if not os.path.exists(recons_dir):
            os.makedirs(recons_dir)

        save_image(recons.data, os.path.join(recons_dir, f"Epoch_{epoch}.png"),
            normalize=True, nrow=12)

        samples = self.model.generate(144, self.device)
        samples_dir = os.path.join(self.log_dir, "Samples")
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        save_image(samples.cpu().data, os.path.join(samples_dir, f"Epoch_{epoch}.png"),
            normalize=True, nrow=12)

    def log_scalar(self, name, value, step):
        if self.log_tool == 'tensorboard':
            self.writer.add_scalar(name, value, step)
        elif self.log_tool == 'wandb':
            self.writer.log({name: value, 'step': step})

    def log_images(self, name, image_tensor, step):
        if self.log_tool == 'tensorboard':
            # Create a grid of images
            grid = vutils.make_grid(image_tensor)

            # Log the grid to TensorBoard
            self.writer.add_image(f'{name} epoch: {step}', grid, global_step=step)
        elif self.log_tool == 'wandb':
            self.writer.log({name: [self.writer.Image(img) for img in image_tensor], 'step': step})

    def resume_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.start_epoch = checkpoint['epoch']
        if self.resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> Resumed optimizer state from checkpoint '{}'".format(ckpt_path))

        else:
            print("=> Not resumed optimizer state.")
        print("=> Resumed checkpoint '{}'".format(ckpt_path))

    def save_checkpoint(self, epoch, filename='checkpoint_latest.pth'):
        state = {
            'state_dict': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }

        model_dir = os.path.join(self.log_dir, 'checkpoints')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        filename = os.path.join(model_dir, filename)
        torch.save(state, filename)
