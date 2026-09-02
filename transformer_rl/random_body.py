"""The **random-design baseline**: the same control stack, with the generator replaced by a coin.

A rival, not an ablation. Experiment 5 asks whether a *learned* designer beats methods that do not
learn one, and the cheapest such method is to redraw the population every window from a fixed
distribution -- the **uniform-size body draw** (`morphology.uniform_size_body`). What it isolates is
the value of the search: control sees the same body variety, on the same budget, through the same
network, and the only thing it never gets is a body chosen because it scored well.

Three things follow from having no generator, and all three are consequences rather than settings:

**Nothing generator-side updates.** `_resample_update` is what fits GenAct, GenCrit and the control
clone, and it is never called -- so the clone's KL and MSE are absent too. They are not "turned
off": they are terms inside the generator's single loss, and with no generator gradient there is
nothing for them to regularize (that is what makes them a *clone*).

**FD and FK stay on.** They are control-side, armed per PPO minibatch in `calc_gradients`, and
belong to the control stack this arm is meant to share with the others.

**The committed body is a measurement, not a read-out.** A learned generator is asked for its best
body; a coin has none, so the run's answer is the best body it happened to *see*. Raw argmax over a
run is unusable -- `num_morphs` defaults to `num_actors`, so every body is one env and one episode,
and the maximum of ~200k single-episode returns is a draw from the noise's upper tail rather than a
good design (experiments/CONTEXT.md, "Selection noise"). So the run keeps a `Shortlist`: the top-K
bodies by raw return, carried in the checkpoint, to be re-evaluated properly afterwards at a real
env count. K is small enough that re-scoring the whole shortlist costs about one window.
"""
from __future__ import annotations

import random
from typing import List, Optional

import torch

from codesigner.interfaces import Morphology, MorphologyGenerator

from .codesign_agent import CodesignAgent
from .algorithm import CodesignAlgorithm
from .morphology import arrays_from_designs, designs_from_arrays, uniform_size_bodies

SHORTLIST_K = 32


class Shortlist:
    """The best `k` bodies a run has seen, by raw per-design return.

    A candidate set for a later re-evaluation, and deliberately not a verdict. Entries are kept in
    the generator's array vocabulary rather than as `Morphology` objects, for the same reason the
    per-window population dump is: that is the representation a checkpoint can carry across a
    process boundary without pickling live library objects, and it round-trips exactly for any body
    the draw produces.
    """

    def __init__(self, k: int = SHORTLIST_K):
        self.k = k
        self.entries: List[dict] = []          # [{score, window, index, body}], best first

    def offer(self, scores, bodies: List[Morphology], window: int) -> None:
        """Merge one window's `(score, body)` pairs in and keep the best `k`."""
        merged = self.entries + [{"score": float(s), "window": int(window), "index": i,
                                  "body": bodies[i]}
                                 for i, s in enumerate(scores.tolist())]
        merged.sort(key=lambda e: e["score"], reverse=True)
        self.entries = merged[:self.k]

    @property
    def best(self) -> Optional[Morphology]:
        return self.entries[0]["body"] if self.entries else None

    def bodies(self) -> List[Morphology]:
        return [e["body"] for e in self.entries]

    def state(self, library) -> dict:
        counts, eff_sub, cap_sub = arrays_from_designs(library, self.bodies(),
                                                       library.max_effectors)
        return {"k": self.k, "counts": counts.to(torch.int16), "eff_sub": eff_sub.to(torch.int16),
                "cap_sub": cap_sub.to(torch.int16),
                "scores": torch.tensor([e["score"] for e in self.entries]),
                "windows": torch.tensor([e["window"] for e in self.entries]),
                "indices": torch.tensor([e["index"] for e in self.entries])}

    def load(self, library, state: dict) -> None:
        self.k = int(state.get("k", self.k))
        bodies = designs_from_arrays(library, state["counts"].long(),
                                     state["eff_sub"].long(), state["cap_sub"].long())
        self.entries = [{"score": float(s), "window": int(w), "index": int(i), "body": b}
                        for s, w, i, b in zip(state["scores"].tolist(),
                                              state["windows"].tolist(),
                                              state["indices"].tolist(), bodies)]


class RandomBodyAgent(CodesignAgent):
    """`CodesignAgent` with the generator taken out of the window boundary.

    Both seams are replaced and nothing else is: the window still ticks on the same cadence, the
    same accumulators reset, the Episode-return carry still happens, and control PPO runs every
    epoch exactly as it does under a learned generator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Both set by `RandomBodyAlgorithm._start`, which owns the run's seed and the shortlist the
        # checkpoint has to carry. Declared here so the attributes exist before the first window.
        self._body_rng: Optional[random.Random] = None
        self._shortlist: Optional[Shortlist] = None

    def _window_update(self, R, obses):
        """Score the window; learn nothing from it.

        `R` is per-ENV and the shortlist is per-DESIGN, so it is folded over each design's env
        block -- the Task lays groups out contiguously, which is the layout `_pin_bodies` relies on
        too. Divided back out of the reward shaper's scale so a shortlist score reads in Episode
        return, the unit the re-evaluation will report in.
        """
        bodies = self._built_morphs                       # what R was just earned on
        per_design = R.view(len(bodies), -1).mean(dim=1) / self._r_scale
        self._shortlist.offer(per_design, bodies, self._gen_window)
        self._gen_log = self._quality_log(
            R, self._win_r_sum * (self._r_scale / max(1, self._win_n_steps)))

    def _next_population(self, N):
        """A fresh uniform-size draw, one body per design group.

        No trace: there is no sampling record because there was no sampler to record, so the
        per-window population dump and the intent-side `build/*` metrics are absent by construction
        rather than empty. The realized ones (`build/*_realized`) still describe this population,
        because they are read off the arrays.
        """
        env = self._env()
        morphs = uniform_size_bodies(self._ml, env.n_morphs, self._body_rng)
        counts, eff_sub, cap_sub = arrays_from_designs(self._ml, morphs, self._max_len,
                                                       self._cur_counts.device)
        per_group = N // env.n_morphs
        if per_group > 1:
            counts = counts.repeat_interleave(per_group, dim=0)
            eff_sub = eff_sub.repeat_interleave(per_group, dim=0)
            cap_sub = cap_sub.repeat_interleave(per_group, dim=0)
        return None, counts, eff_sub, cap_sub, morphs


class RandomMorphologyGenerator(MorphologyGenerator):
    """The uniform-size draw, as a `MorphologyGenerator`.

    `deterministic=False` is the distribution the run actually trained against, which is what makes
    the package's exploration metrics read the right thing for this arm. `deterministic=True` is
    the run's **committed body** -- the shortlist's leader, i.e. the best body this run has *seen*.
    That is the honest analogue of asking a trained generator for its argmax, and it is the body the
    specialization pass fine-tunes.
    """

    def __init__(self, library, shortlist: Shortlist, rng: random.Random,
                 name: str = "uniform_size_draw"):
        super().__init__(name)
        self.library = library
        self.shortlist = shortlist
        self.rng = rng

    def generate(self, n: int, deterministic: bool = False) -> List[Morphology]:
        if not deterministic:
            return uniform_size_bodies(self.library, n, self.rng)
        best = self.shortlist.best
        if best is None:
            raise RuntimeError(
                "this run has no committed body: the shortlist is empty, so no window has closed "
                "and nothing has been scored. A random-design baseline has no body to hand back "
                "before it has measured one.")
        return [best] * n


class RandomBodyAlgorithm(CodesignAlgorithm):
    """The random-design baseline. Same network, config, Task and control stack; no designer."""

    def __init__(self, *args, name: str = "random_design", shortlist_k: int = SHORTLIST_K,
                 **kwargs):
        # A distinct name on purpose: `optimize`'s resume refuses a run directory written by a
        # different algorithm, and a random-design payload continuing a codesign run would leave a
        # directory that reads afterwards as one coherent run.
        super().__init__(*args, name=name, **kwargs)
        self.description = ("control PPO over a population redrawn every window from the "
                            "uniform-size body draw; nothing about the body is learned.")
        self._shortlist = Shortlist(shortlist_k)
        # One stream for the training population and one for anything a caller draws off the
        # artifact, so reading `generate()` from a notebook cannot shift the bodies the run trains
        # on. Seeded from the run's seed, so the population is reproducible.
        seed = self._cfg["params"].get("seed")
        self._body_rng = random.Random(seed)
        self._draw_rng = random.Random(None if seed is None else seed + 1)

    @property
    def shortlist(self) -> Shortlist:
        return self._shortlist

    def _agent_class(self):
        return RandomBodyAgent

    def _initial_bodies(self, base_morphology, num_morphs):
        """Window 0 is drawn like every other window. A codesign run spends window 0 on the seed
        body because its generator has learnt nothing yet; this one has nothing to learn, so
        starting it on the seed would just spend a forty-eighth of the budget off-distribution."""
        return uniform_size_bodies(self.modlib, num_morphs, self._body_rng)

    def _make_generator(self, net, library):
        return RandomMorphologyGenerator(library, self._shortlist, self._draw_rng)

    def _start(self) -> None:
        super()._start()
        self._agent._body_rng = self._body_rng
        self._agent._shortlist = self._shortlist

    # ---- capabilities ----------------------------------------------------------------
    # Both are overridden UNDECORATED, which is how a capability is withdrawn: `capabilities()`
    # scans the class, so the subclass's plain method is what it finds and the algorithm stops
    # advertising something it cannot do. Neither is a stub for later -- there is nothing here to
    # implement. Spread control is an inverse temperature on generator logits and this arm has no
    # logits; design-quality prediction is GenCrit, whose head exists but was never trained, so a
    # metric reading it would be reading noise and reporting it as a belief.

    def spread_at(self, spread: float, n: int):
        raise NotImplementedError(
            "the random-design baseline has no spread control: its draw is at maximum spread by "
            "construction and has no temperature to lower.")

    def predict_design_quality(self, designs):
        raise NotImplementedError(
            "the random-design baseline predicts nothing about a design: GenCrit is never updated "
            "in this arm, so its output is untrained noise rather than a belief.")

    # ---- checkpoints -----------------------------------------------------------------

    def checkpoint_payload(self) -> dict:
        """The weights, plus the shortlist -- which is the run's only trace of which body won.

        Not deferrable to the scrape. The per-window population dump is generator-only and this arm
        writes none, and the metric record lists which bodies were evaluated but never which of
        them scored best. Nothing else survives the process, so a shortlist that was not in the
        checkpoint when the run was launched is a committed body lost.
        """
        return {**super().checkpoint_payload(),
                "shortlist": self._shortlist.state(self.modlib),
                # Both streams' positions. A resumed run whose draw restarted from the seed would
                # re-run a population it had already scored and offer those bodies to the shortlist
                # a second time, at a second noisy score -- which is a duplicate entry in the one
                # structure whose whole job is to rank bodies against each other. Restored before
                # `_start`, so the continuing process draws forward rather than over.
                "body_rng": self._body_rng.getstate(),
                "draw_rng": self._draw_rng.getstate()}

    def load_checkpoint_payload(self, payload: dict) -> None:
        super().load_checkpoint_payload(payload)
        for rng, key in ((self._body_rng, "body_rng"), (self._draw_rng, "draw_rng")):
            if payload.get(key) is not None:
                # torch.save round-trips the tuple's inner tuple as a list; setstate wants tuples.
                version, internal, gauss = payload[key]
                rng.setstate((version, tuple(internal), gauss))
        if payload.get("shortlist") is not None:
            self._shortlist.load(self.modlib, payload["shortlist"])
            if self._agent is not None:
                self._agent._shortlist = self._shortlist
