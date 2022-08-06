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
        self.epochs = epochs
        self.start_epoch = 0
        self.total_steps = 0

        self.print_freq = print_freq
        self.log_dir = log_dir
        self.resume = resume

        if log_tool == 'tensorboard':
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "log"))

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

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
            for epoch in range(self.start_epoch, self.epochs):
                losses_m = utils.AverageMeter()
                for batch_idx, batch in enumerate(tqdm(self.train_dataloader)):
                    self.process_batch(batch, losses_m)

                    if batch_idx % self.print_freq == 0:
                        print("Epoch:{}, batch: {}, loss: {:.5f}".format(epoch, batch_idx, losses_m.avg))
            
                if epoch < self.epochs-1:
                    self.validate(epoch)

            print("=> Traing finished.")
            self.save_checkpoint(self.epochs)
            self.writer.close()

        except KeyboardInterrupt:
            self.save_checkpoint(epoch)

    def process_batch(self, batch, losses_m):
        self.model.train()
        self.optimizer.zero_grad()

        inputs, gt = batch
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, gt.to(self.device))
        self.writer.add_scalar("Train/Loss", loss.item(), self.total_steps)
        losses_m.update(loss.item(), inputs.size(0))
        
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
                inputs, target = batch
                output = self.model(inputs)
                loss = self.criterion(output, target)
                losses_v.update(loss.item(), inputs.size(0))
                self.model.accuracy(output, target)

        self.writer.add_scalar("Valid/Loss", losses_v.avg, epoch)
        self.writer.add_scalar("Valid/Accuracy", self.model.get_test_acc(), epoch)
        print(f'Epoch: {epoch}, validate accuracy: {self.model.get_test_acc()} %')

    def resume_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
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
