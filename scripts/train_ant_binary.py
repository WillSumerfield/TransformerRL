"""PPO training on AntBinaryLegEnv: all-8-leg ant, presence as a continuous probability p.

Experiment env for the value-gradient study (docs/value_gradient_propagation.md): the critic sees
per-leg presence probability p (both length slots), the body is Bernoulli-sampled from p, so V learns
E[return] over p. Reuses the multimorph transformer + masked-norm model unchanged (obs still 139D /
16D actions). Train 3 seeds:
  python scripts/train_ant_binary.py train --seed 42 --name s42
  python scripts/train_ant_binary.py train --seed 43 --name s43
  python scripts/train_ant_binary.py train --seed 44 --name s44
"""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant_binary_leg import AntBinaryLegEnv
from transformer_rl import MultiMorphLegTransformerBuilder

run_training(
    default_config="ppo_ant_binary.yaml",
    train_dir="runs/ant_binary",
    name="ant_binary_transformer",
    env_class=AntBinaryLegEnv,
    env_name="ant-binary-env",
    network=("multimorph_leg_transformer", MultiMorphLegTransformerBuilder),
    model="transformer_masked_a2c_logstd",
)
