#!/usr/bin/env python3
"""
Create reconstructed-space error bands by unfolding and refolding histograms.

This script performs a full unfolding-refolding pipeline to propagate systematic
uncertainties through the reconstruction process:

1. Load migration matrix, reconstruction histogram, and efficiency histogram
2. Unfold the reconstruction histogram to true space using either Moore-Penrose
   pseudoinverse or Tikhonov regularization
3. For each systematic uncertainty universe:
   - Apply unfolding to that universe's reconstruction histogram
   - Refold using that universe's migration matrix and efficiency
4. Propagate all systematic error bands through the unfolding-refolding process
5. Output a new histogram with "R-dagger" errors that represent the
   reconstruction-space uncertainties accounting for unfolding effects

The result enables more accurate error propagation in reco space.

Typical usage:
    python create_reco_errors.py -i selection.root -m migration.root \
        -e efficiency.root -o output_dir/
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
from null_test import flatten_mnvh2d, mnvh2d_from_flattened, print_all, plot_difference

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
TEST_MODE = True


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
    chi2 = np.sum(np.divide(residual**2, T, out=np.zeros_like(residual), where=T > 0))
    return chi2

def main():
    """
    Main pipeline for unfolding-refolding with systematic uncertainty propagation.

    Workflow:
    1. Parse command-line arguments
    2. Load central value histograms: migration matrix (U), reconstruction (R),
       and efficiency (E)
    3. Perform consistency checks on loaded histograms
    4. Unfold R to true space using specified regularization method
    5. Process lateral and vertical error bands:
       - Load universe-specific histograms
       - Unfold each universe
       - Refold to reconstruction space
    6. Save output histogram with refolded error bands

    Returns
    -------
    int
        0 on success, 1 on failure.
    """

    # ========================================================================
    # PARSE ARGUMENTS AND SETUP
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Unfold histograms in true space, then fold back in "
        "systematics universes."
    )
    parser.add_argument(
        "--selection",
        "-i",
        required=True,
        help="Input ROOT file containing selection (reconstruction) output",
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
        "--outdir",
        "-o",
        required=True,
        help="Output directory to save results",
    )
    parser.add_argument(
        "--rcond",
        type=float,
        default=1.0e-6,
        help="Condition number threshold for Moore-Penrose pseudoinverse "
        "(default: 1e-6)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0e-6,
        help="Tikhonov regularization parameter lambda (default: 1e-6)",
    )
    parser.add_argument(
        "--use_mp",
        "-u",
        action="store_true",
        help="Use Moore-Penrose pseudoinverse instead of Tikhonov regularization",
    )
    parser.add_argument(
        "--histo_name",
        "-n",
        default="h_ptrecpsiprime",
        help='Name of histogram to unfold (default: "h_ptrecpsiprime")',
    )
    parser.add_argument(
        "--use_truth",
        action="store_true",
        help="Use truth histogram instead of unfolded reconstruction histogram",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry run without saving output or plots (testing mode)",
    )
    args = parser.parse_args()

    # Create output directories
    outdir = args.outdir if args.outdir else "."
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outdirplots = os.path.join(outdir, "plots")
    if not os.path.exists(outdirplots):
        os.makedirs(outdirplots, exist_ok=True)

    # Assign input files
    selection_file = args.selection
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

    # Load reconstruction histogram R
    print("\n--- Loading reconstruction histogram ---")
    f_in = ROOT.TFile.Open(selection_file, "READ")
    h_reco = PlotUtils.MnvH2D(f_in.Get(args.histo_name + "_qelike"))
    python_utils.disown_mnvh2d(h_reco)
    if not h_reco:
        print(
            f"ERROR: Could not find reco histogram named "
            f"{args.histo_name}_qelike in file {selection_file}"
        )
        sys.exit(1)
    R = flatten_mnvh2d(h_reco)
    print(f"Reconstruction histogram R shape: {R.shape}")

    # Load efficiency-weighted truth ET (numerator of efficiency)
    print("\n--- Loading efficiency-weighted truth ---")
    h_et = PlotUtils.MnvH2D(f_mig.Get(args.histo_name + "_qelike_truth"))
    python_utils.disown_mnvh2d(h_et)
    ET = flatten_mnvh2d(h_et)
    print(f"Efficiency-truth ET shape: {ET.shape}")

    # Load efficiency E = selected/true
    print("\n--- Loading efficiency ---")
    f_eff_in = ROOT.TFile.Open(efficiency_file, "READ")
    h_eff_denominator = PlotUtils.MnvH2D(
        f_eff_in.Get(args.histo_name + "_truth_qelike")
    )
    h_eff_numerator = PlotUtils.MnvH2D(f_eff_in.Get(args.histo_name + "_qelike"))
    h_eff = PlotUtils.MnvH2D(h_eff_numerator.Clone("efficiency"))
    h_eff.Divide(h_eff_numerator, h_eff_denominator)
    python_utils.disown_mnvh2d(h_eff)
    E = flatten_mnvh2d(h_eff)
    print(f"Efficiency E shape: {E.shape}")
    python_utils.disown_mnvh2d(h_eff_numerator)
    num = flatten_mnvh2d(h_eff_numerator)

    # ========================================================================
    # CONSISTENCY CHECKS
    # ========================================================================
    print("\n--- Running consistency checks ---")

    # Check 1: Reconstruction histogram from selection should match
    # reconstruction from migration matrix
    h_reco_mig = PlotUtils.MnvH2D(f_mig.Get(args.histo_name + "_qelike_reco"))
    python_utils.disown_mnvh2d(h_reco_mig)
    R_mig = flatten_mnvh2d(h_reco_mig)
    diff_R = R - R_mig
    mismatches_R = (np.abs(diff_R) > 1.0e-5).sum()
    nonzero_bins_R = (R != 0).sum()
    print(
        f"R_selection vs R_migration: {mismatches_R} mismatches "
        f"out of {len(R)} bins ({nonzero_bins_R} non-zero)"
    )
    if mismatches_R > 0:
        for i, _ in enumerate(diff_R):
            if np.abs(diff_R[i]) > 1.0e-5:
                print(
                    f"  Bin {i}: R_sel={R[i]}, R_mig={R_mig[i]}, "
                    f"diff={diff_R[i]}"
                )

    # Check 2: ET from migration should equal E*T from efficiency
    python_utils.disown_mnvh2d(h_eff_denominator)
    T_eff = flatten_mnvh2d(h_eff_denominator)
    ET_eff = np.multiply(E, T_eff)
    diff_ET = ET - ET_eff
    mismatches_ET = (np.abs(diff_ET) > 1.0e-5).sum()
    nonzero_bins_ET = (ET != 0).sum()
    print(
        f"ET vs E*T: {mismatches_ET} mismatches "
        f"out of {len(ET)} bins ({nonzero_bins_ET} non-zero)"
    )
    if mismatches_ET > 0:
        for i, _ in enumerate(diff_ET):
            if np.abs(diff_ET[i]) > 1.0e-5:
                print(
                    f"  Bin {i}: ET_mig={ET[i]}, E*T_eff={ET_eff[i]}, "
                    f"numerator={num[i]}, truth={T_eff[i]}"
                )

    # Check 3: R should equal U @ ET
    R_from_U_ET = U @ ET
    diff_R_2 = R - R_from_U_ET
    mismatches_UET = (np.abs(diff_R_2) > 1.0e-5).sum()
    print(
        f"R vs U*ET: {mismatches_UET} mismatches "
        f"out of {len(R)} bins ({nonzero_bins_R} non-zero)"
    )
    if mismatches_UET > 0:
        for i, _ in enumerate(diff_R_2):
            if np.abs(diff_R_2[i]) > 1.0e-5:
                print(
                    f"  Bin {i}: R={R[i]}, U*ET={R_from_U_ET[i]}, "
                    f"diff={diff_R_2[i]}"
                )

    # ========================================================================
    # UNFOLD CENTRAL VALUE
    # ========================================================================
    print("\n--- Unfolding central value ---")

    if args.dry_run:
        print("Dry run complete (no unfolding performed).")
        return 0

    if args.use_truth:
        print("Using truth histogram (no unfolding).")
        W_tik = T_eff
    else:
        if args.use_mp:
            # Moore-Penrose pseudoinverse unfolding
            print(
                f"Using Moore-Penrose pseudoinverse (rcond={args.rcond})"
            )
            Uinv, h_pinv, _ = compute_pseudoinverse_from_object(
                h_mig, rcond=args.rcond
            )
            print(
                f"Pseudoinverse shape: {Uinv.shape}, "
                f"matrix shape: {U.shape}"
            )
            print(
                f"Reconstruction test: "
                f"max|U*U^-1*U - U| = {np.abs(U.dot(Uinv).dot(U) - U).max():.3e}"
            )
            W = Uinv @ R
            plot_difference(
                mnvh2d_from_flattened(W, h_reco),
                h_et,
                os.path.join(outdirplots, "residuals_UinvR_minus_ET.png"),
            )
            W_tik = W
        else:
            # Tikhonov regularization unfolding
            print(
                f"Using Tikhonov regularization (lambda={args.lam})"
            )
            W_tik = tikhonov_inverse(
                U, R, lam=args.lam, L=np.identity(U.shape[1])
            )

        # Test refolding without efficiency correction
        R_test_tik = U @ W_tik
        plot_difference(
            mnvh2d_from_flattened(R_test_tik, h_reco),
            h_reco,
            os.path.join(
                outdirplots, "residuals_RTikhonov_minus_R_noEFF.png"
            ),
        )
        R_test_mp = U @ W if args.use_mp else R_test_tik
        plot_difference(
            mnvh2d_from_flattened(R_test_mp, h_reco),
            h_reco,
            os.path.join(
                outdirplots, "residuals_RMoorePenrose_minus_R_noEFF.png"
            ),
        )

        # Apply efficiency correction: W_true = W_reco / E
        W = np.divide(W, E, out=np.zeros_like(W), where=E != 0) if args.use_mp else W_tik
        W_tik = np.divide(
            W_tik, E, out=np.zeros_like(W_tik), where=E != 0
        )
        print("Efficiency correction applied to unfolded spectrum.")

        # Refold for comparison
        EW_tik = np.multiply(W_tik, E)
        R_folded_tik = U @ EW_tik
        plot_difference(
            mnvh2d_from_flattened(R_folded_tik, h_reco),
            mnvh2d_from_flattened(R_test_tik, h_reco),
            os.path.join(outdirplots, "residuals_RTikhonov_minus_R.png"),
        )

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
        list_of_laterror_bands = ["BeamAngleX"]

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
            
            plot_difference(h_R_dagger, h_refolded, os.path.join(outdirplots, f"residuals_CV_refolded_minus_R_{error_band}_{i}.png"))


        h_refolded.PopLatErrorBand(error_band)  # remove any existing error band with the same name
        new_error_band = PlotUtils.MnvLatErrorBand2D(error_band, h_refolded, list_of_hists)
        h_refolded.PushErrorBand(error_band, new_error_band)
        print(f"Pushed new lateral error band {error_band} with {nUniverses} universes to refolded histogram.")

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
            if TEST_MODE and i >= 1:
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
            
            plot_difference(h_R_dagger, h_refolded, os.path.join(outdirplots, f"residuals_CV_refolded_minus_R_{error_band}_{i}.png"))
            
            if TEST_MODE:
                h2d_systErrors_star = h_reco.GetTotalError(False, False, False)
                for i in range(h_reco.GetXaxis().GetNbins()-2, h_reco.GetXaxis().GetNbins()+1):
                    for j in range(1, h_reco.GetYaxis().GetNbins()+1):
                        err_star = h2d_systErrors_star.GetBinContent(i,j)
                        err_dagger = h_R_dagger.GetBinContent(i,j) - h_reco.GetBinContent(i,j)
                        print(f"Bin ({i},{j}): dR* = {err_star} dR+ = {err_dagger}")

        h_refolded.PopVertErrorBand(error_band)  # remove any existing error band with the same name
        new_error_band = PlotUtils.MnvVertErrorBand2D(error_band, h_refolded, list_of_hists)
        h_refolded.PushErrorBand(error_band, new_error_band)
        print(f"Pushed new vertical error band {error_band} with {nUniverses} universes to refolded histogram.")



    if TEST_MODE:
        # compare systematic errors before and after refolding
        h2d_systErrors_star = h_reco.GetTotalError(False, False, False)
        h2d_systErrors_dagger = h_refolded.GetTotalError(False, False, False)
        for i in range(1, h_reco.GetXaxis().GetNbins()+1):
            for j in range(1, h_reco.GetYaxis().GetNbins()+1):
                err_star = h2d_systErrors_star.GetBinContent(i,j)
                err_dagger = h2d_systErrors_dagger.GetBinContent(i,j)
                if ( abs(err_star) > 0. or abs(err_dagger) > 0. ):
                    print(f"Bin ({i},{j}): dR* = {err_star} dR+ = {err_dagger}")

    h_refolded.SetName(h_reco.GetName() + "_refolded")
    output_file_path = os.path.join(outdir, "refolded_errors.root")
    f_out = ROOT.TFile.Open(output_file_path, "RECREATE")

    # Write original reconstruction histogram for comparison
    h_reco.Write()

    # Write refolded histogram with error bands
    h_refolded.Write()

    # Write regularization parameter for documentation
    rcond_hist = ROOT.TH1F("rcond", "rcond", 1, 0, 1)
    rcond_hist.SetBinContent(1, args.rcond)
    rcond_hist.Write()

    f_out.Close()
    print(f"Refolded reco histogram with \"R-dagger\" errors written to {output_file_path}")

    return 0


if __name__ == "__main__":
    """Execute main pipeline when script is run directly."""
    sys.exit(main())
    # main()
