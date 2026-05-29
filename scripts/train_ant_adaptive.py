"""PPO training on AntAdaptiveEnv: multi-morphology env with 3-4 leg morphologies."""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant_adaptive import AntAdaptiveEnv, _MORPHOLOGIES
from transformer_rl import MultiMorphLegTransformerBuilder


def post_config(args, config):
    n_morphs = len(_MORPHOLOGIES)
    n_envs = config["params"]["config"]["num_actors"]
    epm = max(1, n_envs // n_morphs)
    config["params"]["config"]["num_actors"] = n_morphs * epm


run_training(
    default_config="ppo_ant_adaptive.yaml",
    train_dir="runs/ant_adaptive",
    env_class=AntAdaptiveEnv,
    env_name="ant-adaptive-env",
    network=("multimorph_leg_transformer", MultiMorphLegTransformerBuilder),
    post_config_fn=post_config,
)
