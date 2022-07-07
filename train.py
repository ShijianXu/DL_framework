import os
import argparse
import importlib

from numpy import require

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deep Learning Framework", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config", required=True, type=str, help="The config file."
    )
    parser.add_argument(
        "--log", required=True, type=str, help="Path to log."
    )

    args = parser.parse_args()

    config_src = ""
    with open(args.config) as cfg:
        config_src = cfg.read()
        spec = importlib.util.spec_from_loader("config", loader=None)
        config = importlib.util.module_from_spec(spec)
        exec(config_src, config.__dict__)

    print(config.model)

