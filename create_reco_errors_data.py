#!/usr/bin/env python3
"""
Create reconstructed-space error bands for data by unfolding and refolding.

This script performs an unfolding-refolding pipeline specifically for
background-subtracted data histograms to propagate systematic uncertainties
through the reconstruction process. Unlike create_reco_errors.py which works
on simulated events, this operates on real data with background subtraction.

Workflow:
1. Load migration matrix, background-subtracted data histogram, and efficiency
2. Unfold the data histogram to true space using Tikhonov regularization
3. For each systematic uncertainty universe:
   - Apply unfolding to that universe's histogram
   - Refold using that universe's migration matrix and efficiency
4. Propagate all systematic error bands through unfolding-refolding
5. Save new histogram with "R-dagger" errors accounting for unfolding effects
6. Write output back to the input file using UPDATE mode

The result enables proper error propagation for data measurements in
reconstruction space after unfolding effects are accounted for.

Typical usage:
    python create_reco_errors_data.py \\
        --main_file data_analysis.root \\
        --migration migration.root \\
        --efficiency efficiency.root \\
        --lam 1e-6
"""

import os
import sys
import argparse
import numpy as np

# Unfolding and utility functions
from invert_migration_matrix import (
    compute_pseudoinverse_from_object,
    get_th2_from_object,
    matrix_from_th2,
    th2_from_matrix,
    tikhonov_inverse,
)
from null_test import (
    flatten_mnvh2d,
    mnvh2d_from_flattened,
    print_all,
    plot_difference,
)

# ROOT and MINERvA analysis utilities
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.PyConfig.Ownership = False
except Exception as e:
    print("ERROR: PyROOT/ROOT import failed:", e)
    raise

import matplotlib.pyplot as plt
import PlotUtils
import python_utils

# Global flag for testing (limits error band processing when True)
TEST_MODE = False


def compute_chisquare(W, T):
    """
    Compute chi-squared between unfolded (W) and truth (T) histograms.

    Parameters
    ----------
    W : np.ndarray
        Unfolded histogram values.
    T : np.ndarray
        Truth histogram values.

    Returns
    -------
    float
        Sum of (T - W)^2 / T for all bins where T > 0.
    """
    residual = T - W
    chi2 = np.sum(
        np.divide(residual**2, T, out=np.zeros_like(residual), where=T > 0)
    )
    return chi2

def main():
    """
    Main pipeline for data unfolding-refolding with error band propagation.

    Workflow:
    1. Parse command-line arguments
    2. Load central value histograms: migration matrix (U), background-subtracted
       data histogram (R), and efficiency (E)
    3. Unfold R to true space using Tikhonov regularization
    4. Process lateral and vertical error bands:
       - Load universe-specific histograms
       - Unfold each universe
       - Refold to reconstruction space
    5. Save output histogram with refolded error bands back to input file

    Returns
    -------
    int
        0 on success, 1 on failure.
    """

    # ========================================================================
    # PARSE ARGUMENTS AND SETUP
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Unfold background-subtracted data histograms in true "
        "space, then fold back with systematic uncertainties."
    )
    parser.add_argument(
        "--main_file",
        "-i",
        required=True,
        help="Input ROOT file containing all analysis and data output",
    )
    parser.add_argument(
        "--migration",
        "-m",
        required=True,
        help="Input ROOT file containing migration matrix",
    )
    parser.add_argument(
        "--efficiency",
        "-e",
        required=True,
        help="Input ROOT file containing efficiency/purity output",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0e-6,
        help="Tikhonov regularization parameter lambda (default: 1e-6)",
    )
    parser.add_argument(
        "--histo_name",
        "-n",
        default="h_ptrecpsiprime",
        help='Name of histogram to unfold (default: "h_ptrecpsiprime")',
    )
    args = parser.parse_args()

    # Output directory for diagnostic plots
    outdir = "data_errors_refolded"
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # Assign input files
    input_file = args.main_file
    migration_file = args.migration
    efficiency_file = args.efficiency

    # ========================================================================
    # LOAD CENTRAL VALUE HISTOGRAMS
    # ========================================================================

    # Load migration matrix U
    print("\n--- Loading migration matrix ---")
    f_mig = ROOT.TFile.Open(migration_file, "READ")
    h_mig = PlotUtils.MnvH2D(
        f_mig.Get(args.histo_name + "_qelike_migration")
    )
    python_utils.disown_mnvh2d(h_mig)

    if not h_mig:
        print(
            f"ERROR: Could not find migration histogram named "
            f"{args.histo_name}_qelike_migration in file {args.migration}"
        )
        sys.exit(1)

    # Convert to matrix form and normalize by true spectrum (columns)
    U = matrix_from_th2(h_mig)
    Y = np.sum(U, axis=0)  # Sum columns -> true spectrum projection
    X = np.sum(U, axis=1)  # Sum rows -> reconstruction spectrum projection
    U = np.divide(U, Y, out=np.zeros_like(U), where=Y != 0)
    print(f"Migration matrix U converted to array shape: {U.shape}")

    # Load background-subtracted data histogram R
    print("\n--- Loading background-subtracted data histogram ---")
    f_in = ROOT.TFile.Open(input_file, "READ")
    h_reco = PlotUtils.MnvH2D(
        f_in.Get(args.histo_name + "_data_bkgsub")
    )
    python_utils.disown_mnvh2d(h_reco)
    if not h_reco:
        print(
            f"ERROR: Could not find reco histogram named "
            f"{args.histo_name}_data_bkgsub in file {args.main_file}"
        )
        sys.exit(1)
    R = flatten_mnvh2d(h_reco)
    # Set to zero all negative bins (can happen with background subtraction/regularization)
    R = np.where(R > 0, R, 0)
    print(f"Background-subtracted data histogram R shape: {R.shape}")

    # Load efficiency-weighted truth ET (numerator of efficiency)
    print("\n--- Loading efficiency-weighted truth ---")
    h_et = PlotUtils.MnvH2D(f_mig.Get(args.histo_name + "_qelike_truth"))
    python_utils.disown_mnvh2d(h_et)
    ET = flatten_mnvh2d(h_et)
    print(f"Efficiency-truth ET shape: {ET.shape}")

    # Load efficiency E
    print("\n--- Loading efficiency ---")
    f_eff_in = ROOT.TFile.Open(efficiency_file, "READ")
    h_eff = PlotUtils.MnvH2D(
        f_eff_in.Get(args.histo_name + "_eff_qelike")
    )
    python_utils.disown_mnvh2d(h_eff)
    E = flatten_mnvh2d(h_eff)
    print(f"Efficiency E shape: {E.shape}")

    # Load accompanying numerator and truth for reference
    numerator = PlotUtils.MnvH2D(
        f_eff_in.Get(args.histo_name + "_qelike")
    )
    python_utils.disown_mnvh2d(numerator)
    num = flatten_mnvh2d(numerator)

    h_truth_eff = PlotUtils.MnvH2D(
        f_eff_in.Get(args.histo_name + "_truth_qelike")
    )
    python_utils.disown_mnvh2d(h_truth_eff)
    T_eff = flatten_mnvh2d(h_truth_eff)

    # ========================================================================
    # UNFOLD CENTRAL VALUE
    # ========================================================================
    print("\n--- Unfolding central value data ---")
    print(f"Computing Tikhonov regularized inverse with lambda={args.lam}")
    W_tik = tikhonov_inverse(
        U, R, lam=args.lam, L=np.identity(U.shape[1])
    )
    W_tik = np.divide(
        W_tik, E, out=np.zeros_like(W_tik), where=E != 0
    )
    print("Unfolded data histogram W computed.")

    # Set to true spectrum where efficiency or truth is zero (no prediction)
    W_tik = np.where(ET > 0, W_tik, T_eff)

    # ========================================================================
    # INITIALIZE REFOLDED HISTOGRAM
    # ========================================================================
    print("\n--- Processing error bands ---")
    h_refolded = PlotUtils.MnvH2D(h_reco.Clone("R_dagger"))

    # ========================================================================
    # PROCESS LATERAL ERROR BANDS
    # ========================================================================
    print("\n  Lateral error bands:")
    list_of_laterror_bands = h_reco.GetLatErrorBandNames()
    if TEST_MODE:
        list_of_laterror_bands = []

    for error_band in list_of_laterror_bands:
        error_band_obj = h_reco.GetLatErrorBand(error_band)
        nUniverses = error_band_obj.GetNHists()
        print(f"    {error_band}: {nUniverses} universes")
        list_of_hists = []

        for i in range(nUniverses):
            # Build histogram name for this universe
            name = (
                args.histo_name
                + "_qelike_migration_lat_"
                + str(error_band)
                + f"_{i}"
            )

            # Load efficiency for this universe
            h_eff_star = h_eff.GetLatErrorBand(error_band).GetHist(i)
            E_star = flatten_mnvh2d(h_eff_star)

            # Load migration matrix for this universe
            this_mig = f_mig.Get(name)
            if not this_mig:
                print(f"      WARNING: Could not find {name}")
                continue

            # Convert migration matrix and normalize
            U_star = matrix_from_th2(this_mig)
            Y_star = np.sum(U_star, axis=0)
            X_star = np.sum(U_star, axis=1)
            U_star = np.divide(
                U_star, Y_star, out=np.zeros_like(U_star), where=Y_star != 0
            )

            # Refold for this universe: R_dagger = U* @ (E* * W)
            R_dagger = U_star @ (np.multiply(W_tik, E_star))
            h_R_dagger = mnvh2d_from_flattened(R_dagger, h_eff_star)
            list_of_hists.append(h_R_dagger)

            print(f"      Universe {i}: integral = {h_R_dagger.Integral()}")
            plot_difference(
                h_R_dagger,
                h_refolded,
                os.path.join(
                    outdir,
                    f"data_residuals_CV_refolded_minus_R_{error_band}_{i}.png",
                ),
            )

        # Add error band to refolded histogram
        h_refolded.PopLatErrorBand(error_band)
        new_error_band = PlotUtils.MnvLatErrorBand2D(
            error_band, h_refolded, list_of_hists
        )
        h_refolded.PushErrorBand(error_band, new_error_band)
        print(f"    Added {len(list_of_hists)} universes to output")

    # ========================================================================
    # PROCESS VERTICAL ERROR BANDS
    # ========================================================================
    print("\n  Vertical error bands:")
    list_of_verterror_bands = h_reco.GetVertErrorBandNames()
    if TEST_MODE:
        list_of_verterror_bands = ["Flux"]

    for error_band in list_of_verterror_bands:
        error_band_obj = h_reco.GetVertErrorBand(error_band)
        nUniverses = error_band_obj.GetNHists()
        print(f"    {error_band}: {nUniverses} universes")
        list_of_hists = []

        for i in range(nUniverses):
            # Skip additional universes in test mode
            if TEST_MODE and i >= -1:
                continue

            # Build histogram name for this universe
            name = (
                args.histo_name
                + "_qelike_migration_vert_"
                + str(error_band)
                + f"_{i}"
            )

            # Load efficiency for this universe
            h_eff_star = h_eff.GetVertErrorBand(error_band).GetHist(i)
            E_star = flatten_mnvh2d(h_eff_star)

            # Load migration matrix for this universe
            this_mig = f_mig.Get(name)
            if not this_mig:
                print(f"      WARNING: Could not find {name}")
                continue

            # Convert migration matrix and normalize
            U_star = matrix_from_th2(this_mig)
            Y_star = np.sum(U_star, axis=0)
            X_star = np.sum(U_star, axis=1)
            U_star = np.divide(
                U_star, Y_star, out=np.zeros_like(U_star), where=Y_star != 0
            )

            # Refold for this universe: R_dagger = U* @ (E* * W)
            EW_star = np.multiply(W_tik, E_star)
            R_dagger = U_star @ EW_star
            h_R_dagger = mnvh2d_from_flattened(R_dagger, h_eff_star)
            list_of_hists.append(h_R_dagger)

            print(f"      Universe {i}: integral = {h_R_dagger.Integral()}")
            plot_difference(
                h_R_dagger,
                h_refolded,
                os.path.join(
                    outdir,
                    f"data_residuals_CV_refolded_minus_R_{error_band}_{i}.png",
                ),
            )

            # Optional: print bin-by-bin error comparison for debugging
            if TEST_MODE:
                h2d_systErrors_star = h_reco.GetTotalError(
                    False, False, False
                )
                for i in range(
                    h_reco.GetXaxis().GetNbins() - 2,
                    h_reco.GetXaxis().GetNbins() + 1,
                ):
                    for j in range(1, h_reco.GetYaxis().GetNbins() + 1):
                        err_star = h2d_systErrors_star.GetBinContent(i, j)
                        err_dagger = (
                            h_R_dagger.GetBinContent(i, j)
                            - h_reco.GetBinContent(i, j)
                        )
                        print(
                            f"Bin ({i},{j}): dR* = {err_star} dR+ = "
                            f"{err_dagger}"
                        )
                return 0

        # Add error band to refolded histogram
        h_refolded.PopVertErrorBand(error_band)
        new_error_band = PlotUtils.MnvVertErrorBand2D(
            error_band, h_refolded, list_of_hists
        )
        h_refolded.PushErrorBand(error_band, new_error_band)
        print(f"    Added {len(list_of_hists)} universes to output")

    # ========================================================================
    # OPTIONAL: COMPARE SYSTEMATIC ERRORS
    # ========================================================================
    if not TEST_MODE:
        # Compare systematic errors before and after refolding
        h2d_systErrors_star = h_reco.GetTotalError(False, False, False)
        h2d_systErrors_dagger = h_refolded.GetTotalError(False, False, False)
        for i in range(1, h_reco.GetXaxis().GetNbins() + 1):
            for j in range(1, h_reco.GetYaxis().GetNbins() + 1):
                err_star = h2d_systErrors_star.GetBinContent(i, j)
                err_dagger = h2d_systErrors_dagger.GetBinContent(i, j)
                if abs(err_star) > 0.0 or abs(err_dagger) > 0.0:
                    print(f"Bin ({i},{j}): dR* = {err_star} dR+ = {err_dagger}")

    # ========================================================================
    # SAVE OUTPUT
    # ========================================================================
    print("\n--- Writing output ---")

    # Save refolded histogram with error bands back to input file (UPDATE mode)
    output_file = ROOT.TFile.Open(input_file, "UPDATE")
    if not output_file:
        print(f"ERROR: Could not open file {input_file} for writing!")
        sys.exit(1)

    output_file.cd()
    h_refolded.Write(
        args.histo_name + "_data_bkgsub_dagger", ROOT.TObject.kOverwrite
    )
    print(
        f"Saved refolded histogram with systematic error bands to "
        f"{input_file} with name {args.histo_name}_data_bkgsub_dagger"
    )
    output_file.Close()

    return 0

if __name__ == "__main__":
    """Execute main pipeline when script is run directly."""
    sys.exit(main())