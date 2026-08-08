"""The two artifacts a codesign run hands back, as the package defines them (D18).

`CodesignAgent` trains one shared-trunk network carrying both a control policy and a morphology
generator, and drives it directly -- these wrappers are not in that loop. They are what a *user*
takes away: a controller they can step a task with, and a designer they can draw bodies from,
neither of which requires the agent, the optimizer, or rl_games.

Both hold the live network rather than a copy, so an artifact handed out mid-run keeps tracking the
run. That is what makes `optimize`'s "best so far" bookkeeping cheap, and it is why a caller who
wants a frozen artifact should checkpoint instead.
"""
from __future__ import annotations

import torch

from codesigner.interfaces import ControlPolicy, MorphologyGenerator

from .morphology import designs_from_arrays


class TransformerControlPolicy(ControlPolicy):
    """The shared-trunk transformer's control head, as a steppable policy.

    Wraps the rl_games *model* (not the bare network) so observation normalization -- including the
    raw `{0,1}` tail that must survive it -- goes through exactly the path training used. A policy
    that renormalized differently from its training would score its own checkpoint wrong.
    """

    def __init__(self, model, name: str = "transformer_control"):
        super().__init__(name)
        self.model = model

    @torch.no_grad()
    def act(self, observation, deterministic: bool = True):
        # eval() every call, not once in __init__: the caller may hold this artifact across
        # training, where the agent puts the model back in train mode, and a normalizer that
        # updates its statistics on evaluation observations corrupts the run it came from.
        was_training = self.model.training
        self.model.eval()
        try:
            res = self.model({"is_train": False, "prev_actions": None, "obs": observation})
        finally:
            self.model.train(was_training)
        actions = res["mus"] if deterministic else res["actions"]
        # The task masks inactive DOFs itself, but out-of-range actions on the active ones are not
        # its problem -- rl_games clamps in `preprocess_actions`, and stepping a task directly
        # (evaluate) has no such step in between.
        return actions.clamp(-1.0, 1.0)


class TransformerMorphologyGenerator(MorphologyGenerator):
    """The generator head, as a batched distribution over bodies.

    `generate` walks the same frontier MDP training samples from, so a drawn body is drawn from the
    distribution that was actually optimized -- not a re-derivation of it.
    """

    def __init__(self, net, library, name: str = "transformer_generator"):
        super().__init__(name)
        self.net = net
        self.library = library

    @torch.no_grad()
    def generate(self, n: int, deterministic: bool = False) -> list:
        # 'greedy' is the argmax of the masked token policy at every frontier step, which is the
        # generator's committed design. It is deterministic in the tokens, but the MDP visits
        # growable limbs in a random order, so n>1 greedy draws are near-identical rather than
        # bit-identical -- the ordering only matters where two limbs are still tied.
        trace = self.net.sample(n, mode="greedy" if deterministic else "stochastic")
        return designs_from_arrays(self.library, trace["counts"].long(),
                                   trace["eff_sub"], trace["cap_sub"])
