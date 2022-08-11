import os
import torch
from tqdm import tqdm
import utils
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

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
        print_freq=400,
        log_dir='./logs',
        resume=True,
        log_tool='tensorboard'
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.start_epoch = 0
        self.total_steps = 0

        self.print_freq = print_freq
        self.log_dir = log_dir
        self.resume = resume

        # init logger
        if log_tool == 'tensorboard':
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "log"))

        # check device
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # to device
        self.model.to(self.device)
        self.criterion.to(self.device)

    def train(self):
        # from pudb import set_trace; set_trace()

        if self.resume:
            ckpt_path = os.path.join(self.log_dir, 'checkpoints', 'checkpoint_latest.pth')
            if os.path.exists(os.path.join(self.log_dir, 'checkpoints')) and os.path.isfile(ckpt_path):
                    print("=> Resuming ckpt ...")
                    self.resume_ckpt(ckpt_path)
            else:
                print("=> No ckpt in log dir.")
        else:
            print("=> Start training ...")

        try:
            self.losses_m = utils.AverageMeter()
            for epoch in range(self.start_epoch, self.epochs):
                for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                    self.process_batch(batch)

                    if batch_idx % self.print_freq == 0:
                        print("Epoch: {}, batch: {}, train loss: {:.5f}".format(epoch, batch_idx, self.losses_m.avg))
            
                self._on_epoch_end(epoch)
                self.losses_m.reset()

            print("=> Traing finished.")
            self.save_checkpoint(self.epochs)
            self.writer.close()

        except KeyboardInterrupt:
            self.save_checkpoint(epoch)

    def _on_epoch_end(self, epoch):
        if epoch < self.epochs-1:
            valid_loss = self.validate(epoch)
            print("Epoch: {}, valid loss: {:.5f}".format(epoch, valid_loss))

            if self.scheduler is not None:
                # self.scheduler.step(valid_loss)
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("Train/Learning rate", lr, epoch)

    def process_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        source, target = batch
        source = source.to(self.device)
        target = target.to(self.device)
        output = self.model(source)

        # The returned loss is a dict
        losses = self.model.compute_loss(source, output, target, self.criterion)        
        self.log_scalar(losses, self.total_steps, prefix='Train')

        self.losses_m.update(losses['loss'].item(), source.size(0))
        
        losses['loss'].backward()
        self.optimizer.step()
        self.total_steps += 1

    def log_scalar(self, s_dict, step, prefix='Train'):
        for item in s_dict:
            item_name = prefix + '/' + item
            self.writer.add_scalar(item_name, s_dict[item].item(), step)

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
                losses = self.model.compute_loss(source, output, target, self.criterion)                 
                losses_v.update(losses['loss'].item(), source.size(0))

                # accuracy for classification
                # PSNR for dense prediction
                self.model.compute_metric(source, output, target)

        self.writer.add_scalar("Valid/Loss", losses_v.avg, epoch)
        self.writer.add_scalar("Valid/Metric", self.model.get_metric_value(), epoch)

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

        save_image(recons.data,
            os.path.join(recons_dir,
            f"Epoch_{epoch}.png"),
            normalize=True,
            nrow=12)

        samples = self.model.sample(144, self.device)
        samples_dir = os.path.join(self.log_dir, "Samples")
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        save_image(samples.cpu().data,
            os.path.join(samples_dir,
            f"Epoch_{epoch}.png"),
            normalize=True,
            nrow=12)

    def resume_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
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
