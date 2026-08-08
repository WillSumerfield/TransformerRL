"""PPG training on AntCodesignEnv (disjoint policy/value nets; see ppg_agent.py).

Same env/network as train_ant_full.py; algo.name=ppg_continuous in the config
selects PPGAgent, which builds the value net internally (multimorph_limb_value).
"""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from codesigner.components.tasks import Ant
from transformer_rl import MultiMorphLimbTransformerBuilder

run_training(
    default_config="ppo_ant_ppg.yaml",
    train_dir="runs/ant_ppg",
    name="ant_ppg_transformer",
    env_class=Ant,
    env_name="ant-codesign-env",
    network=("multimorph_limb_transformer", MultiMorphLimbTransformerBuilder),
    model="transformer_masked_a2c_logstd",
)
