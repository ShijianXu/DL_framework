import argparse
import importlib

from trainers import Trainer
from tester import Tester

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deep Learning Framework", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--mode", type=str, default="train", help="Select train/test model."
    )
    parser.add_argument(
        "--config", required=True, type=str, help="The config file."
    )
    parser.add_argument(
        "--log", type=str, default='./logs', help="Path to log."
    )
    parser.add_argument(
        "--auto_resume", type=bool, default=True, help="Whether to check and load the latest checkpoint from log dir."
    )
    parser.add_argument(
        "--ckpt", type=str, help="Path to the test checkpoint."
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

    if args.mode == 'train':
        trainer = Trainer(
            config=config,
            model=config.model,
            train_dataloader=config.train_dataloader,
            val_dataloader=config.test_dataloader,
            criterion=config.loss,
            optimizer=config.optimizer,
            epochs=config.num_epochs,
            print_freq=args.print_freq,
            log_dir=args.log,
            resume=args.auto_resume,
        )
        trainer.train()

    else:
        tester = Tester(
            config=config,
            model=config.model,
            dataloader=config.test_dataloader,
            ckpt_path=args.ckpt,
        )
        tester.test()