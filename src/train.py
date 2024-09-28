from llamafactory.train.tuner import run_exp
from typing import Optional, Dict, Any


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
