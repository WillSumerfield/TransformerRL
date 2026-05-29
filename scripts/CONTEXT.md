# Training

PPO training, Optuna tuning, and play/render orchestration for the leg transformer over the ant envs. Owns `scripts/` and `configs/`. Each `train_ant_*.py` pairs with a `configs/ppo_*.yaml`; configs select the registered network/model by name and the env by `env_name`.

## Language

This context introduces no vocabulary of its own — it composes the shared kernel ([Context Map](../CONTEXT-MAP.md)) with the [Control](../transformer_rl/CONTEXT.md) and [Morphology](../envs/CONTEXT.md) contexts, plus standard RL/PPO/Optuna terms. Add terms here only if a usage diverges from their ordinary meaning.
