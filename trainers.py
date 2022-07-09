import os
import torch
from tqdm import tqdm
import utils

class Trainer(object):
    def __init__(self,
        model,
        dataloader,
        criterion,
        optimizer,
        epochs
    ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self):
        for epoch in range(self.epochs):
            losses_m = utils.AverageMeter()
            for batch_idx, batch in enumerate(self.dataloader, 0):
                self.process_batch(batch, losses_m)

                if batch_idx % 400 == 0:
                    print("Epoch:{}, batch: {}, loss: {:.5f}".format(epoch, batch_idx, losses_m.avg))

    def process_batch(self, batch, losses_m):
        self.model.train()
        self.optimizer.zero_grad()

        inputs, gt = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, gt)
        losses_m.update(loss.item(), inputs.size(0))
        
        loss.backward()
        self.optimizer.step()

        
