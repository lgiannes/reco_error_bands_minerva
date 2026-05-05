# reco_error_bands_minerva

Python utilities to **propagate model/systematic uncertainties into reconstructed-space (reco) histograms** by
**unfolding to true space and refolding back to reco space** for every systematic universe.

This repo is tailored to MINERvA-style analyses using **ROOT + PlotUtils (MnvH2D)** error bands (lateral/vertical)
and produces a “refolded” reco histogram whose error bands incorporate the unfolding/refolding response.

## What problem this solves

In many workflows, uncertainties are evaluated in reco space directly (error bands on a reco histogram `R`). Interaction models uncertainties are implemented by varying the truth space histograms,[...]

These scripts build a reco-space histogram with more correct error bars, often referred to here as **R† (“R-dagger”)**.
Starting from:
- a migration/response matrix **U** (reco × true) with corresponding error band universes **U***
- an efficiecny histogram **E** with corresponding error band universes **E***
  
The error propagation is done in one of two ways:

1. If the truth CV histogram **T** is available, propagate the errors to reco space by multiplying the truth CV with the E and U systematic universes:
   
  `R† = U* · (E* ⊙ T)`

  Where `⊙`  is elementwise multiplication

2. If the truth CV histogram is not available, unfold the reco hsitogram **R** to a true-space estimate **W**, then for each systematic universe `*`, refold using that universe’s response/efficiency:

  `R† = U* · (E* ⊙ W)`


The output is a reco-space histogram with systematic error bands that have been propagated through the
unfold/refold procedure. The ``traditionally'' propagated histograms (`R† = U* · (E* ⊙ T*)`) are also saved to the output file, so the user can chose to use them instead. This can be useful f[...]

## Contents

- `create_reco_errors.py` — main MC-style workflow: unfold a reco histogram and refold all systematic universes
- `create_reco_errors_data.py` — data workflow: same idea but for a background-subtracted data histogram
- `invert_migration_matrix.py` — utilities to extract matrices from ROOT histograms and compute inverses
  - Moore–Penrose pseudoinverse (`numpy.linalg.pinv`)
  - Tikhonov regularization solver
- `run_create_reco_errors.sh`, `run_create_reco_errors_data.sh` — example launch scripts (paths are site/user-specific)
- `null_test.py` — quick null/sanity test to validate your environment + refold/unfold plumbing
- `python_utils.py` — shared helper utilities used across scripts (I/O, histogram helpers, plotting, convenience functions)

## Requirements

These scripts assume an analysis environment that provides:

- Python 3
- `numpy`, `matplotlib`
- **PyROOT/ROOT**
- **PlotUtils** (MINERvA)

## Inputs (ROOT objects expected)

The scripts expect MINERvA-style `MnvH2D` objects with specific naming conventions.
For the default `--histo_name h_ptrecpsiprime`:

### Migration file (`--migration`)

- `${histo_name}_qelike_migration` (2D migration matrix)
- `${histo_name}_qelike_truth` (truth distribution used for checks)
- `${histo_name}_qelike_reco` (reco distribution used for checks)
- Universe-specific migration matrices named like:
  - `${histo_name}_qelike_migration_lat_<BandName>_<i>`
  - `${histo_name}_qelike_migration_vert_<BandName>_<i>`

### Selection / reco file (`--selection`, MC workflow)

- `${histo_name}_qelike` (reco histogram with error bands)

### Efficiency file (`--efficiency`)

`create_reco_errors.py` builds efficiency from numerator/denominator:

- `${histo_name}_qelike` (selected)
- `${histo_name}_truth_qelike` (truth)

`create_reco_errors_data.py` expects a precomputed efficiency histogram:

- `${histo_name}_eff_qelike`

### Data file (`--main_file`, data workflow)

- `${histo_name}_data_bkgsub` (background-subtracted data histogram with error bands)

## How it works (high level)

1. **Load** central-value migration matrix `U`, reco histogram `R`, and efficiency `E`.
2. **Normalize** migration matrix columns so each true bin forms `P(reco | true)`.
3. **Unfold**:
   - Either Moore–Penrose pseudoinverse (`--use_mp`, with `--rcond`), or
   - Tikhonov regularization (default, with `--lam`).
4. **Efficiency-correct** unfolded spectrum in true space (`W / E`).
5. **Loop over error bands** on the input reco histogram (`GetLatErrorBandNames()`, `GetVertErrorBandNames()`):
   - Pull the universe-specific efficiency `E_*` from the efficiency histogram error bands
   - Pull the universe-specific migration matrix `U_*` from the migration file
   - Compute refolded universe `R†_*` and build a new `Mnv(H|Lat|Vert)ErrorBand2D`
6. **Write outputs**:
   - `create_reco_errors.py`: writes `refolded_errors.root` containing the original reco hist and the refolded one
   - `create_reco_errors_data.py`: writes back into the input file as `${histo_name}_data_bkgsub_dagger`

## Usage

### MC / simulation-style

```bash
python3 create_reco_errors.py \
  --migration Migration.root \
  --selection Selection.root \
  --efficiency Efficiency.root \
  --outdir out/ \
  --lam 1e-6
```

Options of note:

- `--use_mp` / `--rcond <float>`: use Moore–Penrose pseudoinverse unfolding
- `--lam <float>`: Tikhonov regularization strength (default path)
- `--use_truth`: skip unfolding and use the truth histogram (debug/closure)
- `--dry_run`: parse and load inputs only (no output)

### Data (background-subtracted)

```bash
python3 create_reco_errors_data.py \
  --main_file Analysis_histograms.root \
  --migration Migration.root \
  --efficiency Efficiency.root \
  --lam 1e-6
```

## Quick validation

Run the null test to make sure your environment and MAT/ROOT bindings are set up correctly:

```bash
python3 null_test.py
```

## Notes / caveats

- I haven't used create_reco_errors_data.py in my analysis, so that should require further validation from future users. Although the structure is identical to create_reco_errors.py
- The code currently has a `TEST_MODE` flag (see `create_reco_errors.py`) that restricts processed error bands.
  Set to `False` to process all bands.
- Some bin-indexing in `invert_migration_matrix.py` uses ROOT bin numbering directly; double-check indexing
  if you modify `matrix_from_th2` / `th2_from_matrix`.
- Paths in the `run_*.sh` scripts are examples for an EOS-based environment and will need adaptation.

## Citation / context

This workflow is common when you want reco-space uncertainties that already incorporate the response
effects implicit in unfolding/refolding, and when you are using `MnvH2D` systematic universes.
