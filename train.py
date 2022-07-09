import argparse
import importlib

from trainers import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deep Learning Framework", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config", required=True, type=str, help="The config file."
    )
    parser.add_argument(
        "--log", type=str, default='./logs', help="Path to log."
    )
    parser.add_argument(
        "--resume", type=str, help="Path to the resumed checkpoint."
    )
    parser.add_argument(
        "--print_freq", type=int, default=400, help="Loss print frequency."
    )

    args = parser.parse_args()

    config_src = ""
    with open(args.config) as cfg:
        config_src = cfg.read()
        spec = importlib.util.spec_from_loader("config", loader=None)
        config = importlib.util.module_from_spec(spec)
        exec(config_src, config.__dict__)

    trainer = Trainer(
        config=config,
        model=config.model,
        dataloader=config.train_dataloader,
        criterion=config.loss,
        optimizer=config.optimizer,
        epochs=config.num_epochs,
        print_freq=args.print_freq,
        log_dir=args.log,
    )
    trainer.train()
