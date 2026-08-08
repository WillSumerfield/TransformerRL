"""Single-network codesign: control + morphology generator on one shared trunk (CodesignAgent,
codesign_tokens). See temp/codesign_single_network_plan.md."""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from codesigner.components.tasks import Ant
from transformer_rl import MultiMorphLimbTransformerBuilder

run_training(
    default_config="ppo_ant_codesign_single.yaml",
    train_dir="runs/ant_codesign",
    name="ant_codesign_single_transformer",
    env_class=Ant,
    env_name="ant-codesign-env",
    network=("multimorph_limb_transformer", MultiMorphLimbTransformerBuilder),
    model="transformer_masked_a2c_logstd",
)
