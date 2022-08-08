import os
import torch
from tqdm import tqdm
import utils
from torch.utils.tensorboard import SummaryWriter

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
                        print("Epoch:{}, batch: {}, loss: {:.5f}".format(epoch, batch_idx, self.losses_m.avg))
            
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

            if self.scheduler is not None:
                self.scheduler.step(valid_loss)

    def process_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        source, target = batch
        outputs = self.model(source.to(self.device))
        loss = self.criterion(outputs, target.to(self.device))
        self.writer.add_scalar("Train/Loss", loss.item(), self.total_steps)
        self.losses_m.update(loss.item(), source.size(0))
        
        loss.backward()
        self.optimizer.step()
        self.total_steps += 1

    def validate(self, epoch):
        self.model.eval()
        self.model.reset_counter()
        losses_v = utils.AverageMeter()
        print("=> Validating ...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader)):
                source, target = batch
                output = self.model(source.to(self.device))
                loss = self.criterion(output, target.to(self.device))
                losses_v.update(loss.item(), source.size(0))

                # accuracy for classification
                # PSNR for dense prediction
                self.model.compute_metric(epoch, output, target.to(self.device))

        self.writer.add_scalar("Valid/Loss", losses_v.avg, epoch)
        self.writer.add_scalar("Valid/Metric", self.model.get_metric_value(), epoch)
        return losses_v.avg

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
