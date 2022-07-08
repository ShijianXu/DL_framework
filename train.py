import argparse
import importlib


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
            for i, data in enumerate(self.dataloader, 0):
                self.process_batch(data)

    def process_batch(self, batch):
        inputs, labels = batch
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        print(loss.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deep Learning Framework", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config", required=True, type=str, help="The config file."
    )
    parser.add_argument(
        "--log", type=str, help="Path to log."
    )

    args = parser.parse_args()

    config_src = ""
    with open(args.config) as cfg:
        config_src = cfg.read()
        spec = importlib.util.spec_from_loader("config", loader=None)
        config = importlib.util.module_from_spec(spec)
        exec(config_src, config.__dict__)

    trainer = Trainer(
        model=config.model,
        dataloader=config.train_dataloader,
        criterion=config.loss,
        optimizer=config.optimizer,
        epochs=config.num_epochs
    )
    trainer.train()
