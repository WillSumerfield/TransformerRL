import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant import AntEnv

run_training(
    default_config="ppo_ant_mlp.yaml",
    train_dir="runs/ant",
    env_class=AntEnv,
    env_name="ant-env",
)
