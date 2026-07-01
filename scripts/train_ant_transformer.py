import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant import AntEnv
from transformer_rl import LimbTransformerBuilder


def add_args(parser):
    parser.add_argument("--games_num", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--kl_threshold", type=float)
    parser.add_argument("--deterministic", choices=["True", "False"], default=None)
    parser.add_argument("--experiment_name", type=str)


def post_config(args, config):
    cfg = config["params"]["config"]
    if args.games_num is not None:
        cfg["player"]["games_num"] = args.games_num
    if args.learning_rate is not None:
        cfg["learning_rate"] = args.learning_rate
    if args.kl_threshold is not None:
        cfg["kl_threshold"] = args.kl_threshold
    if args.deterministic is not None:
        cfg["player"]["deterministic"] = args.deterministic.lower() == "true"
    if args.experiment_name is not None:
        cfg["full_experiment_name"] = args.experiment_name


run_training(
    default_config="ppo_ant.yaml",
    train_dir="runs/ant",
    name="ant_transformer",
    env_class=AntEnv,
    env_name="ant-env",
    network=("limb_transformer", LimbTransformerBuilder),
    extra_args_fn=add_args,
    post_config_fn=post_config,
)
