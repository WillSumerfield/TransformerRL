"""Shared measurement layer for the paper experiments: the pieces every experiment script and
scripts/eval.py read from, so a metric has exactly one implementation.

  diversity.py    morphology distances (d_comp/d_struct) + N_modes
  committance.py  typed population representations + entropy-decomposition committance (rho, N_*)
  policy.py       checkpoint -> (net, obs normalizer)

Launch/scrape/eval-pass/stats/artifact modules land here as the first experiment needs them.
"""
