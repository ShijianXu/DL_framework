import argparse
import importlib

from trainers import Trainer
from tester import Tester
from sampler import Sampler

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
        "--resume_optimizer", type=bool, default=True, help="Whether to resume the optimizer state."
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
            criterion=config.loss,
            optimizer=config.optimizer,
            epochs=config.num_epochs,
            callbacks=config.callbacks,
            val_dataloader=config.valid_dataloader if hasattr(config, 'valid_dataloader') else None,
            scheduler=config.scheduler if hasattr(config, 'scheduler') else None,
            eval_metric=config.eval_metric if hasattr(config, 'eval_metric') else None,
            sample_valid=config.sample_valid if hasattr(config, 'sample_valid') else False,
            sample_valid_freq=config.sample_valid_freq if hasattr(config, 'sample_valid_freq') else -1,
            print_freq=args.print_freq,
            log_dir=args.log,
            resume=args.auto_resume,
            resume_optimizer=args.resume_optimizer,
            log_tool='tensorboard', # options: tensorboard, wandb
        )
        trainer.train()

    elif args.mode == 'test':
        tester = Tester(
            config=config,
            model=config.model,
            dataloader=config.test_dataloader,
            ckpt_path=args.ckpt,
        )
        tester.test()

    elif args.mode == 'sample':
        sampler = Sampler(
            config=config,
            model=config.model,
            latent_dim=config.latent_dim,
            ckpt_path=args.ckpt,
        )
        sampler.generate(sample_num=144)

    else:
        raise NotImplementedError("Test model not supported!")