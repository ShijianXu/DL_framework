import os
import torch
from tqdm import tqdm
import utils

class Trainer(object):
    def __init__(self,
        config,
        model,
        dataloader,
        criterion,
        optimizer,
        epochs,
        print_freq=400,
        log_dir='./logs'
    ):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

        self.print_freq = print_freq
        self.log_dir = log_dir

    def train(self):
        for epoch in range(self.epochs):
            losses_m = utils.AverageMeter()
            for batch_idx, batch in enumerate(tqdm(self.dataloader)):
                self.process_batch(batch, losses_m)

                if batch_idx % self.print_freq == 0:
                    print("Epoch:{}, batch: {}, loss: {:.5f}".format(epoch, batch_idx, losses_m.avg))
            
            self.validate()

        print("Traing finished.")
        self.save_checkpoint()

    def process_batch(self, batch, losses_m):
        self.model.train()
        self.optimizer.zero_grad()

        inputs, gt = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, gt)
        losses_m.update(loss.item(), inputs.size(0))
        
        loss.backward()
        self.optimizer.step()

    def validate(self):
        pass

    def save_checkpoint(self, filename='checkpoint_latest.pth'):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        
        filename = os.path.join(self.log_dir, filename)
        torch.save(state, filename)
