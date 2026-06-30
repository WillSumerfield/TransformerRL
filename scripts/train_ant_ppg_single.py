"""Single-network PPG training on AntMultiMorphEnv (one shared trunk; see ppg_agent.py
sec 3.6 path). Same env/network/agent as train_ant_ppg.py; the config's ppg.shared_trunk=true
makes PPGAgent build one net and detach the value gradient at the trunk in the policy phase.
"""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant_multimorph import AntMultiMorphEnv
from transformer_rl import MultiMorphLegTransformerBuilder

run_training(
    default_config="ppo_ant_ppg_single.yaml",
    train_dir="runs/ant_ppg_single",
    name="ant_ppg_single_transformer",
    env_class=AntMultiMorphEnv,
    env_name="ant-multimorph-env",
    network=("multimorph_leg_transformer", MultiMorphLegTransformerBuilder),
    model="transformer_masked_a2c_logstd",
)
