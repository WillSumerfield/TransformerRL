"""PPO training on AntMultiMorphEnv: one EnvironmentGroup per stable morphology."""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant_multimorph import AntMultiMorphEnv
from transformer_rl import MultiMorphLegTransformerBuilder

# Full ant samples num_envs variable-length bodies (one per env) via env.sample_morphs in the config;
# no fixed morphology_set, so num_actors stays as configured (epm=1).
run_training(
    default_config="ppo_ant_full.yaml",
    train_dir="runs/ant_full",
    name="ant_full_transformer",
    env_class=AntMultiMorphEnv,
    env_name="ant-multimorph-env",
    network=("multimorph_leg_transformer", MultiMorphLegTransformerBuilder),
    model="transformer_masked_a2c_logstd",
)
