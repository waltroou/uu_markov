"""Simulates the mock lineage-traced single-cell RNA-seq data for the
seven-state PC9 tumor-heterogeneity notebook.

Instructor-facing. Students get the CSV this writes, not this file: it
contains the true transition matrices they are supposed to estimate.

    python3 instructor/simulate_data.py
        # writes ../cells.csv (student copy) and cells_with_truth.csv (here)
    python3 instructor/simulate_data.py --imbalanced
        # same model, rare resistant cells: ../cells_imbalanced.csv and
        # cells_with_truth_imbalanced.csv

Story
-----
PC9 lung-cancer cells (EGFR-mutant) carry a heritable lineage tag. We
sequence a culture at time point A, treat it for one cycle with one drug,
and sequence it again at time point B. As in ../markov-drug-resistance/,
every tag maps one cell at A to at most one cell at B: a tag seen at A but
missing at B means the cell died.

Drug A is osimertinib (an EGFR inhibitor). Drug B is docetaxel
(chemotherapy that kills cells as they divide).

Cell states (the clusters students should recover). P, NCP and CP use the
marker genes of the matching states in the real PC9 data (Oren et al. 2021,
../markov-persisters-pc9/):
    S-G1   sensitive, resting in G1 (slow-cycling)
    S-syn  sensitive, copying DNA (S phase)
    S-G2M  sensitive, dividing (G2/M)
    P      early drug-tolerant persister
    NCP     non-cycling persister
    CP      cycling persister: keeps dividing on osimertinib
    MDR    multidrug-resistant: drug-efflux pumps (ABCB1/ABCG2) high
Dead cells are never sequenced, so "cleared" shows up only as a missing tag.

Resistance is pre-existing, never induced: no S, P or NCP cell ever becomes CP
or MDR. Persisters are tolerant, not resistant: they survive longer and can
switch back to sensitive states, but under either drug they eventually die.
Collateral sensitivity: CP cells keep dividing, so docetaxel kills them; MDR
cells pump docetaxel out but still depend on EGFR, so osimertinib kills them.
So every one-state tumor can be eliminated by one of the two drugs, while a
tumor holding both CP and MDR cells survives either drug alone.

The CP/MDR pairing is a teaching simplification, not a measured result.
"""

import numpy as np
import pandas as pd

SEED = 5030
STATES = ["S-G1", "S-syn", "S-G2M", "P", "NCP", "CP", "MDR"]
OUTCOMES = STATES + ["cleared"]

# True one-cycle transition probabilities (rows: state at A, cols: outcome at B).
#        S-G1  S-syn S-G2M P     NCP    CP     MDR   cleared
TRUE_P = {
    "osimertinib": np.array([
        [0.10, 0.03, 0.02, 0.20, 0.05, 0.00, 0.00, 0.60],   # S-G1: most likely to persist
        [0.05, 0.08, 0.05, 0.10, 0.02, 0.00, 0.00, 0.70],   # S-syn
        [0.05, 0.05, 0.08, 0.07, 0.00, 0.00, 0.00, 0.75],   # S-G2M
        [0.10, 0.03, 0.02, 0.35, 0.15, 0.00, 0.00, 0.35],   # P: tolerant, reversible
        [0.05, 0.00, 0.00, 0.10, 0.50, 0.00, 0.00, 0.35],   # NCP: tolerant, slow to die
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],   # CP: resistant (absorbing)
        [0.05, 0.03, 0.02, 0.05, 0.00, 0.00, 0.15, 0.70],   # MDR: still EGFR-dependent
    ]),
    "docetaxel": np.array([
        [0.30, 0.10, 0.02, 0.10, 0.10, 0.00, 0.00, 0.38],   # S-G1: not dividing, spared
        [0.05, 0.10, 0.15, 0.05, 0.05, 0.00, 0.00, 0.60],   # S-syn
        [0.02, 0.02, 0.05, 0.02, 0.02, 0.00, 0.00, 0.87],   # S-G2M: killed in mitosis
        [0.10, 0.02, 0.02, 0.40, 0.10, 0.00, 0.00, 0.36],   # P
        [0.05, 0.00, 0.00, 0.10, 0.45, 0.00, 0.00, 0.40],   # NCP
        [0.05, 0.03, 0.05, 0.02, 0.00, 0.10, 0.00, 0.75],   # CP: dividing, so killed
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],   # MDR: resistant (absorbing)
    ]),
}

# This folder's data (cells_with_holiday.csv) was made with a docetaxel matrix in
# which the cells docetaxel spares sometimes switch on drug pumps (see
# add_docetaxel_induction.py): S-G1 -> MDR 0.04, P -> MDR 0.03, NCP -> MDR 0.03.
for _state, _rate in {"S-G1": 0.04, "P": 0.03, "NCP": 0.03}.items():
    _i = STATES.index(_state)
    TRUE_P["docetaxel"][_i, OUTCOMES.index("MDR")] += _rate
    TRUE_P["docetaxel"][_i, OUTCOMES.index("cleared")] -= _rate

# Starting composition of the culture at time point A (same for both arms).
START_MIX = [0.25, 0.15, 0.10, 0.15, 0.10, 0.10, 0.15]
N_CELLS_A = 3000
# A more realistic untreated tumor, for `--imbalanced`: mostly sensitive cells,
# few persisters, and rare resistant cells (about 60 CP and 30 MDR per culture).
START_MIX_IMBALANCED = [0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01]

# ---- Expression model -------------------------------------------------------
# Cell-cycle markers are standard; P, NCP and CP markers are taken from the real
# PC9 states; MDR markers are drug-efflux and detox genes. The data is simulated.
MARKERS = {
    "S-G1":  ["CCND1", "CDK6", "MYC", "E2F1", "CDT1"],
    "S-syn": ["PCNA", "MCM2", "MCM3", "TYMS", "RRM2"],
    "S-G2M": ["MKI67", "TOP2A", "CCNB1", "CDK1", "CENPF"],
    "P":     ["NEAT1", "FILIP1L", "IL32", "GPRC5A", "TNNT1"],
    "NCP":    ["HLA-A", "HLA-B", "TACSTD2", "CD9", "SLPI"],
    "CP":     ["GSTM3", "ALDH3A1", "FN1", "SPP1", "CDA"],
    "MDR":   ["ABCB1", "ABCG2", "ABCC1", "GSTP1", "MT2A"],
}
HOUSEKEEPING = ["ACTB", "GAPDH", "RPLP0", "EEF1A1", "TPT1",
                "MALAT1", "FTL", "TMSB4X", "UBC", "PPIA"]
GENES = [g for s in STATES for g in MARKERS[s]] + HOUSEKEEPING
SENSITIVE = ["S-G1", "S-syn", "S-G2M"]

# ---- Decoy phenotypes -------------------------------------------------------
# Real-looking gene programs that do NOT change how a cell responds to either
# drug. They are invisible at K = 7 and claim one extra cluster each as
# students raise K: K = 8 splits S-G1 by time point (batch), K = 9 adds an
# NCP+IFN cluster, K = 10 adds a stress cluster mixing the sensitive states. They are drawn from a separate random stream, so the 45 genes above
# are identical with or without them.
#   stress  dissociation stress, in a random 12% of all cells, any state
#   ifn     interferon response, in 40% of NCP cells only
#   batch   ribosomal genes higher in every cell sequenced at time point B
DECOYS = {
    "stress": ["FOS", "JUN", "EGR1", "HSPA1A", "DUSP1"],
    "ifn":    ["ISG15", "IFIT1", "IFIT3", "MX1", "OAS1"],
    "batch":  ["RPL13", "RPS19", "RPS27A", "RPL32", "RPS12"],
}
DECOY_GENES = [g for d in DECOYS.values() for g in d]
STRESS_RATE, IFN_RATE_IN_NC = 0.12, 0.40
DECOY_EFFECT = np.log(12.0)       # tuned so K=8 adds batch, K=9 IFN, K=10 stress
BATCH_SHIFT = np.log(3.0)         # ribosomal genes at time point B, all cells


def expression(states, rng, timepoint):
    """Poisson counts from a log-normal mean: baseline + state program."""
    n = len(states)
    log_mu = np.full((n, len(GENES)), np.log(1.0))
    hk = [GENES.index(g) for g in HOUSEKEEPING]
    log_mu[:, hk] = np.log(8.0)
    for i, s in enumerate(states):
        for g in MARKERS[s]:
            log_mu[i, GENES.index(g)] = np.log(10.0)
        # Neighbouring cell-cycle phases share a little of each other's program,
        # and CP cells (which divide) carry a weak S-phase signal.
        if s in SENSITIVE:
            k = SENSITIVE.index(s)
            for other in SENSITIVE[max(k - 1, 0):k + 2]:
                if other != s:
                    for g in MARKERS[other][:2]:
                        log_mu[i, GENES.index(g)] = np.log(2.5)
        if s == "CP":
            for g in MARKERS["S-syn"][:2]:
                log_mu[i, GENES.index(g)] = np.log(2.5)
    log_mu += rng.normal(0, 0.35, size=log_mu.shape)            # cell-to-cell noise
    log_mu += np.log(rng.lognormal(0, 0.25, size=(n, 1)))      # library size
    if timepoint == "B":
        log_mu += 0.08                                         # mild batch effect
    return rng.poisson(np.exp(log_mu))


def decoy_expression(meta, rng):
    """Counts for the decoy genes, plus each cell's (stress, ifn) flags."""
    n = len(meta)
    stress = rng.random(n) < STRESS_RATE
    ifn = (meta["true_state"].to_numpy() == "NCP") & (rng.random(n) < IFN_RATE_IN_NC)
    log_mu = np.full((n, len(DECOY_GENES)), np.log(1.0))
    col = {g: k for k, g in enumerate(DECOY_GENES)}
    for flag, name in [(stress, "stress"), (ifn, "ifn")]:
        for g in DECOYS[name]:
            log_mu[flag, col[g]] = DECOY_EFFECT
    batch = [col[g] for g in DECOYS["batch"]]
    log_mu[:, batch] = np.log(6.0)
    log_mu[np.ix_(meta["timepoint"].to_numpy() == "B", batch)] += BATCH_SHIFT
    log_mu += rng.normal(0, 0.35, size=log_mu.shape)
    return rng.poisson(np.exp(log_mu)), stress, ifn


def simulate(arm, rng, mix=START_MIX):
    P = TRUE_P[arm]
    start = rng.choice(len(STATES), size=N_CELLS_A, p=mix)
    nxt = np.array([rng.choice(len(OUTCOMES), p=P[s]) for s in start])
    prefix = "o" if arm == "osimertinib" else "d"
    tags = [f"tag{prefix}-{i:05d}" for i in rng.permutation(N_CELLS_A)]

    rows = []
    for tag, s in zip(tags, start):
        rows.append(("A", tag, STATES[s]))
    for tag, o in zip(tags, nxt):
        if OUTCOMES[o] != "cleared":
            rows.append(("B", tag, OUTCOMES[o]))
    meta = pd.DataFrame(rows, columns=["timepoint", "lineage_tag", "true_state"])
    X = expression(meta["true_state"].tolist(), rng, None)
    X[meta["timepoint"].values == "B"] = expression(
        meta.loc[meta.timepoint == "B", "true_state"].tolist(), rng, "B")
    df = pd.concat([meta, pd.DataFrame(X, columns=GENES)], axis=1)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "cell_id", [f"cell_{prefix}_{i:05d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    here = Path(__file__).resolve().parent
    imbalanced = "--imbalanced" in sys.argv[1:]
    mix = START_MIX_IMBALANCED if imbalanced else START_MIX
    suffix = "_imbalanced" if imbalanced else ""
    for arm, P in TRUE_P.items():
        assert np.allclose(P.sum(axis=1), 1), arm
    rng = np.random.default_rng(SEED)
    rng_decoy = np.random.default_rng(SEED + 1)
    parts = []
    for arm in ["osimertinib", "docetaxel"]:
        df = simulate(arm, rng, mix)
        df.insert(1, "experiment", arm)
        parts.append(df)
        print(arm, df.shape, df.groupby("timepoint").size().to_dict())
    df = pd.concat(parts, ignore_index=True)
    counts, stress, ifn = decoy_expression(df, rng_decoy)
    df = pd.concat([df, pd.DataFrame(counts, columns=DECOY_GENES)], axis=1)
    truth = df.assign(decoy_stress=stress, decoy_ifn=ifn)
    truth.to_csv(here / f"cells_with_truth{suffix}.csv", index=False)   # instructor copy
    df.drop(columns="true_state").to_csv(here.parent / f"cells{suffix}.csv", index=False)
