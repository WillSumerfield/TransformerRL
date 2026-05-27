"""PPO training on AntCodesignEnv: one EnvironmentGroup per stable morphology."""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant_codesign import AntCodesignEnv, _stable_morphologies
from transformer_rl import DynamicLegTransformerBuilder


def post_config(args, config):
    n_morphs = len(_stable_morphologies())
    n_envs = config["params"]["config"]["num_actors"]
    epm = max(1, n_envs // n_morphs)
    config["params"]["config"]["num_actors"] = n_morphs * epm


run_training(
    default_config="ppo_ant_dynamic_p1.yaml",
    train_dir="runs/ant_dynamic_p1",
    env_class=AntCodesignEnv,
    env_name="ant-codesign-env",
    network=("dynamic_leg_transformer", DynamicLegTransformerBuilder),
    post_config_fn=post_config,
)
