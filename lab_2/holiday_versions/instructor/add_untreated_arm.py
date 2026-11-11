"""Appends an untreated ("drug holiday") control culture to the existing
two-drug data, leaving every existing row unchanged.

    python3 instructor/add_untreated_arm.py
        # reads ../cells.csv and cells_with_truth.csv
        # writes ../cells_with_holiday.csv and cells_with_truth_holiday.csv

The new culture uses the same states, starting mix, expression model and
decoys as simulate_data.py, with its own random streams, so the osimertinib
and docetaxel rows are byte-for-byte the same as before.

Untreated matrix: with no drug, cells cycle (G1 -> S -> G2/M -> G1) and
rarely die (0.05); persisters drift back to sensitive states; CP and MDR
cells partly revert. Same matrix as ../markov-pc9-sequencing-sim/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import simulate_data as sd

#        S-G1  S-syn S-G2M P     NCP    CP     MDR   cleared
UNTREATED_P = np.array([
    [0.40, 0.45, 0.10, 0.00, 0.00, 0.00, 0.00, 0.05],   # S-G1: enters S phase
    [0.10, 0.35, 0.50, 0.00, 0.00, 0.00, 0.00, 0.05],   # S-syn: moves on to G2/M
    [0.55, 0.10, 0.30, 0.00, 0.00, 0.00, 0.00, 0.05],   # S-G2M: divides, back to G1
    [0.40, 0.10, 0.05, 0.35, 0.05, 0.00, 0.00, 0.05],   # P: reverts to sensitive
    [0.20, 0.05, 0.00, 0.25, 0.45, 0.00, 0.00, 0.05],   # NCP: reverts, slowly
    [0.05, 0.05, 0.05, 0.00, 0.00, 0.80, 0.00, 0.05],   # CP: mostly stays
    [0.30, 0.05, 0.05, 0.00, 0.00, 0.00, 0.55, 0.05],   # MDR: pumps switch off
])
assert np.allclose(UNTREATED_P.sum(axis=1), 1)

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    old = pd.read_csv(here.parent / "cells.csv")
    old_truth = pd.read_csv(here / "cells_with_truth.csv")

    sd.TRUE_P["untreated"] = UNTREATED_P
    rng = np.random.default_rng(sd.SEED + 2)
    new = sd.simulate("untreated", rng)            # prefix "u": tags tagu-..., ids cell_u_...
    new.insert(1, "experiment", "untreated")
    counts, stress, ifn = sd.decoy_expression(new, np.random.default_rng(sd.SEED + 3))
    new = pd.concat([new, pd.DataFrame(counts, columns=sd.DECOY_GENES)], axis=1)
    new_truth = new.assign(decoy_stress=stress, decoy_ifn=ifn)
    print("untreated", new.shape, new.groupby("timepoint").size().to_dict())

    truth = pd.concat([old_truth, new_truth[old_truth.columns]], ignore_index=True)
    cells = pd.concat([old, new.drop(columns="true_state")[old.columns]], ignore_index=True)
    truth.to_csv(here / "cells_with_truth_holiday.csv", index=False)
    cells.to_csv(here.parent / "cells_with_holiday.csv", index=False)
    print("wrote", cells.shape)
