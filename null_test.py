#!/usr/bin/env python3
"""
Perform null-test validation for unfolding and refolding consistency.

This script validates the migration-matrix workflow by:
1. Loading reconstruction, truth, efficiency, and migration histograms
2. Flattening 2D histograms into 1D arrays for matrix operations
3. Checking core consistency identities:
    - R ~= R_sel
    - ET ~= E * T
    - R ~= U * ET
4. Computing a Moore-Penrose pseudoinverse for U
5. Unfolding R to W, then refolding back to reco space
6. Comparing residuals and saving diagnostic plots and ROOT outputs

Expected behavior:
- Reco-space quantities should be reproduced within numerical precision.
- Deviations typically indicate binning mismatches, normalization issues,
  or conditioning/regularization effects.

The script operates on MnvH2D/TH2 histograms and converts to NumPy arrays for
linear algebra. It produces both PNG diagnostics and a ROOT output file.
"""

import os
import sys
import argparse
import numpy as np
import python_utils
from invert_migration_matrix import compute_pseudoinverse_from_object, get_th2_from_object, matrix_from_th2, th2_from_matrix
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    # ROOT.gSystem.Load("libPlotUtils.so")
except Exception as e:
    print("ERROR: PyROOT/ROOT import failed:", e)
    raise
import matplotlib.pyplot as plt


def flatten_mnvh2d(h2d):
    """
    Flatten a 2D MnvH2D histogram into a 1D numpy array.

    The vector includes underflow/overflow bins and uses ROOT global bin
    indexing via FindBin(center_x, center_y) to preserve compatibility with
    matrix operations performed elsewhere in the workflow.

    Parameters
    ----------
    h2d : ROOT.TH2 or PlotUtils.MnvH2D
        Input 2D histogram to flatten.

    Returns
    -------
    np.ndarray
        Flattened 1D array containing all bins, including under/overflow.
    """
    nx = h2d.GetNbinsX()
    ny = h2d.GetNbinsY()
    nbins = (h2d.GetNbinsX() + 2) * (h2d.GetNbinsY() + 2) # include under/overflow
    arr = np.zeros((nbins,))

    for ix in range(0, nx + 2):
        for iy in range(0, ny + 2):
            bin_content = h2d.GetBinContent(ix, iy)
            bin_center_x = h2d.GetXaxis().GetBinCenter(ix)
            bin_center_y = h2d.GetYaxis().GetBinCenter(iy)
            bin_index_2d = h2d.FindBin(bin_center_x, bin_center_y) 
            # print(f"ix={ix}, iy={iy} => bin_index_2d={bin_index_2d}, content={bin_content}")
            arr[bin_index_2d] = bin_content  
    return arr

def mnvh2d_from_flattened(arr, template_h2d):
    """
    Create a MnvH2D histogram from a flattened 1D numpy array,
    using the binning of the provided template MnvH2D histogram.

    Parameters
    ----------
    arr : np.ndarray
        Flattened histogram values indexed using ROOT global bin indexing.
    template_h2d : ROOT.TH2 or PlotUtils.MnvH2D
        Template histogram providing shape, binning, and metadata.

    Returns
    -------
    ROOT.TH2 or PlotUtils.MnvH2D
        Cloned histogram filled with values from arr.
    """
    h2d = template_h2d.Clone(template_h2d.GetName() + "_from_flattened")
    ROOT.SetOwnership(h2d, False)
    h2d.Reset()

    nx = h2d.GetNbinsX()
    ny = h2d.GetNbinsY()

    for ix in range(0, nx + 2):
        for iy in range(0, ny + 2):
            bin_index_2d = h2d.GetBin(ix, iy)
            bin_content = arr[bin_index_2d]
            h2d.SetBinContent(ix, iy, bin_content)
    return h2d


def print_all(h1,h2):
    """
    Print bin-by-bin differences between two 2D histograms.

    Parameters
    ----------
    h1 : ROOT.TH2 or PlotUtils.MnvH2D
        First histogram.
    h2 : ROOT.TH2 or PlotUtils.MnvH2D
        Second histogram.
    """
    for ix in range(0, h1.GetNbinsX()+2):
        for iy in range(0, h1.GetNbinsY()+2):
            bin_content_1 = h1.GetBinContent(ix, iy)
            bin_content_2 = h2.GetBinContent(ix, iy)
            if bin_content_1 != bin_content_2:
                print(f"Bin ({ix}, {iy}): H1 = {bin_content_1}, H2 = {bin_content_2}, \t\tdiff = {bin_content_1 - bin_content_2}")

def plot_difference(h2d_1, h2d_2, output_name):
    """
    Compare two 2D histograms and save residual diagnostics.

    This function computes absolute and relative residuals in flattened space,
    prints summary statistics, and saves:
    - A 1D residual distribution PNG
    - A 2D residual heatmap PNG (suffix: _h.png)

    Parameters
    ----------
    h2d_1 : ROOT.TH2 or PlotUtils.MnvH2D
        Reference histogram.
    h2d_2 : ROOT.TH2 or PlotUtils.MnvH2D
        Histogram to compare against reference.
    output_name : str
        Output path for the 1D residual plot.
    """

    dof = 0
    for ix in range(0, h2d_1.GetNbinsX()+2):
        for iy in range(0, h2d_1.GetNbinsY()+2):
            bin_content = h2d_1.GetBinContent(ix, iy)
            if bin_content > 0 :
                dof += 1

    print(f"\nComparing {h2d_1.GetName()} and {h2d_2.GetName()}")
    vec1 = flatten_mnvh2d(h2d_1)
    vec2 = flatten_mnvh2d(h2d_2)
    diff = vec1 - vec2
    rel_diff = np.divide(diff, vec1, out=np.zeros_like(diff), where=(vec1>1.e-4) & (vec2>1.e-4))
    print(f"   Maximum absolute difference: {np.max(np.abs(diff))}")
    print(f"   Maximum relative difference: {np.max(np.abs(rel_diff))}")
    print(f"   RMS of relative difference: {np.sqrt(np.mean(rel_diff**2))}")
    chi2 = np.sum((rel_diff)**2)
    print(f"   chisquare of absolute difference: {chi2:.2e}/{dof}")  
    
    # plot the relative residuals    
    plt.figure(figsize=(10,6))
    plt.hist(rel_diff[np.abs(rel_diff) > 1.e-6], alpha=0.7, label='Residuals', bins=500)
    plt.xlabel('Difference')    
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Relative ' + output_name.replace(".png", "").split("/")[-1])
    plt.legend()
    plt.grid()
    # print on canvas number of rel_diff != 0
    n_nonzero = np.sum(abs(rel_diff) > 1.e-6)
    plt.text(0.7, 0.8, f'Non-zero residuals (>1e-6): {n_nonzero}/{len(rel_diff)}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.7, f'chisquare = {chi2:.2e}/{dof}', transform=plt.gca().transAxes)
    plt.savefig(output_name)

    # now the TH2D
    residual_h2d = mnvh2d_from_flattened(rel_diff, h2d_1)
    canvas_resid = ROOT.TCanvas("c_resid", "c_resid", 800, 600)
    residual_h2d.Draw("colz")
    residual_h2d.SetStats(0)
    canvas_resid.SaveAs(output_name.replace(".png", "_h.png"))
    canvas_resid.Clear()
    canvas_resid.Close()



def main():
    """
    Run the end-to-end null test for unfolding/refolding validation.

    Workflow:
    1. Parse CLI arguments and resolve default input files
    2. Load required histograms from migration/efficiency/selection files
    3. Flatten histograms and normalize migration matrix
    4. Run consistency checks and produce diagnostic plots
    5. Compute pseudoinverse and unfold reco distribution
    6. Refold and compare with original reco distribution
    7. Save outputs as PNG diagnostics and ROOT histograms
    """

    # ====================================================================
    # PARSE ARGUMENTS AND PREPARE OUTPUT
    # ====================================================================
    p = argparse.ArgumentParser(description="Invert migration matrix via Moore-Penrose pseudoinverse")
    p.add_argument('--migration', '-m', default=None, help='Input ROOT file with migration matrices histograms')
    p.add_argument('--effpurity', '-e', default=None, help='Input ROOT file with efficiency histograms')
    p.add_argument('--selection', '-s', default=None, help='Input ROOT with selection histograms')
    p.add_argument('--rcond', type=float, default=1.e-6, help='rcond parameter passed to numpy.linalg.pinv')
    p.add_argument('--histo_name', '-n', default='h_ptrecpsiprime', help='Name of the histogram to unfold (default: "")')
    p.add_argument('--outdir', '-o', default=None, help='Output directory to save histograms (default: current directory)')
    args = p.parse_args()

    outdir = args.outdir if args.outdir else "."
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print("\n=== NULL TEST ===\n")

    # ====================================================================
    # RESOLVE INPUT FILES
    # ====================================================================
    # Set default file paths if not provided
    if args.migration is None:
        args.migration = os.path.join(os.getenv("MINERVA"), "PsiPrimeAnalysisOutputs_minervame1M", "Migration_light.root")
    if args.effpurity is None:
        args.effpurity = os.path.join(os.getenv("MINERVA"), "PsiPrimeAnalysisOutputs_minervame1M", "Efficiency_light.root")
    if args.selection is None:
        args.selection = os.path.join(os.getenv("MINERVA"), "PsiPrimeAnalysisOutputs_minervame1M", "Selection_Signal_light.root")

    # ====================================================================
    # LOAD INPUT HISTOGRAMS
    # ====================================================================
    # Load reco histogram
    f_mig = ROOT.TFile(args.migration)
    h_reco = f_mig.Get(args.histo_name + "_qelike_reco")
    h_true_selected = f_mig.Get(args.histo_name + "_qelike_truth")
    if not h_reco or not h_true_selected:
        print(f"ERROR: Could not find histograms named {args.histo_name}_qelike_reco or {args.histo_name}_qelike_truth in file {args.migration}")
        sys.exit(1)
    # Load reco histogram also from selection file (it should be the same)
    f_sel = ROOT.TFile(args.selection)
    h_reco_sel = f_sel.Get(args.histo_name + "_qelike")
    if not h_reco_sel:
        print(f"ERROR: Could not find histogram named {args.histo_name}_qelike in file {args.selection}")
        sys.exit(1)

    # Load migration matrix
    h_mig = f_mig.Get(args.histo_name + "_qelike_migration")
    if not h_mig:
        print(f"ERROR: Could not find migration histogram named {args.histo_name}_qelike_migration in file {args.migration}")
        sys.exit(1)

    # Load efficiency histogram
    f_eff = ROOT.TFile(args.effpurity)
    h_eff = f_eff.Get(args.histo_name + "_eff_qelike") 
    if not h_eff:
        print(f"ERROR: Could not find efficiency histogram named {args.histo_name}_eff_qelike in file {args.effpurity}")
        sys.exit(1)
    # Load true histogram. This is ACTUAL true, meaning before selection cuts. 
    #                      While h_true_selected is after selection cuts, there is an efficiency correction between them.
    h_true = f_eff.Get(args.histo_name + "_truth_qelike")
    if not h_true:
        print(f"ERROR: Could not find true histogram named {args.histo_name}_truth_qelike in file {args.effpurity}")
        sys.exit(1)

    # Create a dictionary of all histograms for easy access
    histos = {
        'R': h_reco, # from migration file
        'R_sel': h_reco_sel, # from selection file
        'ET': h_true_selected, # T_sel_i = E_i * T_i (from migration file)
        'T': h_true, # from efficiency file
        'U': h_mig, # migration matrix from migration file
        'E': h_eff # efficiency from efficiency file
    }

    # C_R_syst = h_reco.GetSysErrorMatrix()
    C_R_stat = h_reco.GetStatErrorMatrix()
    # C_R_syst = np.array(C_R_syst)
    C_R_stat = np.array(C_R_stat)
    # print(f"Reco histogram R systematic covariance matrix shape: {C_R_syst.shape}")
    print(f"Reco histogram R statistical covariance matrix shape: {C_R_stat.shape}")

    # h_reco_minus_sel = h_reco.Clone("R_minus_Rsel")
    # h_reco_minus_sel.Add(h_reco_sel, -1)
    # canvas = ROOT.TCanvas("c1", "c1", 800, 600)
    # h_reco_minus_sel.Draw("colz")
    # canvas.SaveAs(os.path.join(outdir, "R_minus_Rsel.png"))
    # h_reco.Draw("colz")
    # canvas.SaveAs(os.path.join(outdir, "R_original.png"))
    # h_reco_sel.Draw("colz")
    # canvas.SaveAs(os.path.join(outdir, "R_selection.png"))

    print("Loaded all histograms successfully.")

    # ====================================================================
    # FLATTEN HISTOGRAMS AND BUILD MATRICES
    # ====================================================================
    # Flatten reco histogram to 1D numpy array
    R = flatten_mnvh2d(histos['R'])
    print(f"Flattened reco histogram R to array of shape: {R.shape}")

    # Flatten reco histogram from selection file to 1D numpy array
    R_sel = flatten_mnvh2d(histos['R_sel'])
    print(f"Flattened reco histogram from selection file R_sel to array of shape: {R_sel.shape}")

    # Flatten true selected histogram to 1D numpy array
    ET = flatten_mnvh2d(histos['ET'])
    print(f"Flattened true selected histogram ET to array of shape: {ET.shape}")

    # Flatten true histogram to 1D numpy array
    T = flatten_mnvh2d(histos['T'])
    print(f"Flattened true histogram T to array of shape: {T.shape}")

    # Flatten efficiency histogram to 1D numpy array
    E = flatten_mnvh2d(histos['E'])
    print(f"Flattened efficiency histogram E to array of shape: {E.shape}")

    # Convert migration matrix to 2D numpy array
    U = matrix_from_th2(histos['U'])
    # normalize U so that each column sums to 1
    Y = np.sum(U, axis=0) # sum over axis 0 means summing columns means projection on Y (true)
    X = np.sum(U, axis=1) # sum over axis 1 means summing rows means projection on X (reco)
    U = np.divide(U, Y, out=np.zeros_like(U), where=Y!=0)
    print(f"Converted migration matrix U to array of shape: {U.shape}")

    # (for debugging) save to file R, ET and U as numpy arrays
    np.savez(os.path.join(outdir, "null_test_migration_histograms.npz"),
             R=R,
             ET=ET,
             U=U,
             C_R_stat=C_R_stat)

    # ====================================================================
    # CONSISTENCY CHECKS AND DIAGNOSTICS
    # ====================================================================
    # Basic consistency checks
    # R original and after flattening should be the same
    # R_after_flattening = mnvh2d_from_flattened(R, histos['R'])
    # print("Comparing original R histogram and R histogram after flattening and unflattening:")
    # print_all(histos['R'], R_after_flattening)
    # R_sel = R
    diff_Rsel = np.subtract(R, R_sel, out=np.zeros_like(R), where=(R!=0) | (R_sel!=0))
    rel_diff_Rsel = np.divide(diff_Rsel, R, out=np.zeros_like(diff_Rsel), where=R!=0)
    print(f"Maximum absolute difference between R and R_sel: {np.max(np.abs(diff_Rsel))}")
    print(f"Maximum relative difference between R and R_sel: {np.max(np.abs(rel_diff_Rsel))}")
    print(f"chisquare of absolute difference between R and R_sel: {np.sum((diff_Rsel)**2)}/{len(R)}")  # add small number to avoid division by zero
    plot_difference(histos['R'], histos['R_sel'], os.path.join(outdir, "R_minus_Rsel.png"))
    # loop through the bins and print those that differ
    # for i in range(histos['R'].GetNbinsX()+2):
    #     for j in range(histos['R'].GetNbinsY()+2):
    #         bin_content_R = histos['R'].GetBinContent(i, j)
    #         bin_content_Rsel = histos['R_sel'].GetBinContent(i, j)
    #         print(f"Bin ({i}, {j}): R = {bin_content_R}, R_sel = {bin_content_Rsel}, \t\tdiff = {bin_content_R - bin_content_Rsel}")
    # plot the residuals    
    plt.figure(figsize=(10,6))
    plt.hist(diff_Rsel, alpha=0.7, label='Residuals (R - R_sel)', bins=500)
    plt.xlabel('Difference')    
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Residuals between Reco Histogram R and Selection Reco Histogram R_sel')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outdir, "residuals_R_minus_Rsel.png"))
    # ET = E * T
    ET_test = np.multiply(E, T)
    diff_ET = np.subtract(ET, ET_test, out=np.zeros_like(ET), where=(ET!=0) | (ET_test!=0))
    rel_diff_ET = np.divide(diff_ET, ET, out=np.zeros_like(diff_ET), where=ET!=0)
    print(f"Maximum absolute difference between original ET and E*T: {np.max(np.abs(diff_ET))}")
    print(f"Maximum relative difference between original ET and E*T: {np.max(np.abs(rel_diff_ET))}")
    print(f"chisquare of absolute difference between original ET and E*T: {np.sum((diff_ET)**2)}/{len(ET)}")  
    plot_difference(histos['ET'], mnvh2d_from_flattened(ET_test, histos['ET']), os.path.join(outdir, "ET_minus_ETest.png"))
    
    # plot the residuals    
    plt.figure(figsize=(10,6))
    plt.hist(diff_ET, alpha=0.7, label='Residuals (ET - E*T)', bins=500)
    plt.xlabel('Difference')    
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Residuals between True Selected Histogram ET and E*T')
    n_nonzero = np.sum(abs(rel_diff_ET) > 1.e-6)
    plt.text(0.7, 0.8, f'Non-zero residuals (>1e-6): {n_nonzero}/{len(rel_diff_ET)}', transform=plt.gca().transAxes)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outdir, "residuals_ET_minus_ETest.png"))
    # R = U * ET
    R_test = U @ ET
    # difference (ignore if both are zero)
    diff_R = np.subtract(R, R_test, out=np.zeros_like(R), where=(R!=0) | (R_test!=0))
    rel_diff_R = np.divide(diff_R, R, out=np.zeros_like(diff_R), where=R!=0)
    print(f"Maximum absolute difference between original R and U*ET: {np.max(np.abs(diff_R))}")
    print(f"Maximum relative difference between original R and U*ET: {np.max(np.abs(rel_diff_R))}")
    print(f"chisquare of absolute difference between original R and U*ET: {np.sum((diff_R)**2)}/{len(R)}")  # add small number to avoid division by zero
    plot_difference(histos['R'], mnvh2d_from_flattened(R_test, histos['R']), os.path.join(outdir, "R_minus_UT.png"))
    # plot the residuals
    plt.figure(figsize=(10,6))
    plt.hist(diff_R, alpha=0.7, label='Residuals (R - U*ET)', bins=500)
    plt.xlabel('Difference')    
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Residuals between Original Reco Histogram R and U*ET')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outdir, "residuals_R_minus_UT.png"))
    for i,d in enumerate(R):
        if d > 0:
            diff = R_test[i] - d
            if abs(diff) > 1e-5:
                print(f"R = U*ET test failed at index {i}: R={d}, U*ET={R_test[i]}, diff={diff}")
    # draw projection of migration matrix
    h_mig_px = histos['U'].ProjectionX("h_mig_px")
    h_mig_py = histos['U'].ProjectionY("h_mig_py")
    canvas_mig = ROOT.TCanvas("c_mig", "c_mig", 800, 600)
    h_mig_px.Draw("hist")
    canvas_mig.SaveAs(os.path.join(outdir, "migration_matrix_projectionX.png"))
    h_mig_py.Draw("hist")
    canvas_mig.SaveAs(os.path.join(outdir, "migration_matrix_projectionY.png"))
    # draw R as 1D histogram
    h_R_1d = ROOT.TH1D("h_R_1d", "Reco Histogram R as 1D;Bin;Counts", len(R), 0, len(R))
    for i in range(len(R)):
        h_R_1d.SetBinContent(i, R[i])
    canvas_R_1d = ROOT.TCanvas("c_R_1d", "c_R_1d", 800, 600)
    h_R_1d.Draw("hist")
    canvas_R_1d.SaveAs(os.path.join(outdir, "R_1d_histogram.png"))
    # draw ET as 1D histogram
    h_ET_1d = ROOT.TH1D("h_ET_1d", "True Selected Histogram ET as 1D;Bin;Counts", len(ET), 0, len(ET))
    for i in range(len(ET)):
        h_ET_1d.SetBinContent(i, ET[i])
    canvas_ET_1d = ROOT.TCanvas("c_ET_1d", "c_ET_1d", 800, 600)
    h_ET_1d.Draw("hist")
    canvas_ET_1d.SaveAs(os.path.join(outdir, "ET_1d_histogram.png"))
    # draw R_test as 1D histogram
    h_R_test_1d = ROOT.TH1D("h_R_test_1d", "Reconstructed R_test = U*ET as 1D;Bin;Counts", len(R_test), 0, len(R_test))
    for i in range(len(R_test)):
        h_R_test_1d.SetBinContent(i, R_test[i])
    canvas_R_test_1d = ROOT.TCanvas("c_R_test_1d", "c_R_test_1d", 800, 600)
    h_R_test_1d.Draw("hist")
    canvas_R_test_1d.SaveAs(os.path.join(outdir, "R_test_1d_histogram.png"))

    # ====================================================================
    # PSEUDOINVERSE AND UNFOLDING TESTS
    # ====================================================================
    # Compute pseudoinverse of migration matrix
    print("Computing pseudoinverse of migration matrix...")
    pinv, h_pinv, _ = compute_pseudoinverse_from_object(histos['U'], rcond=args.rcond)
    print(f"Pseudoinverse computed. Shape of pseudoinverse: {pinv.shape}. Shape of migration matrix: {U.shape}")
    # identity matrix diagnostic
    I = np.eye(U.shape[0])
    resid = U.dot(pinv) - I
    count_non_ones = 0
    for i_diag in range(resid.shape[0]):    
        if(resid[i_diag, i_diag] != 0):
            count_non_ones += 1
            # print(f"Residue diag {i_diag}: {resid[i_diag, i_diag]}") 
    print(f"Diagnostic: Number of non-zero elements on diagonal of U*U^-1 - I: {count_non_ones} out of {resid.shape[0]}")
    print(f"Diagnostic: max|U*U^-1*U - U| = {np.abs(U.dot(pinv).dot(U) - U).max():.3e}")
    nx = histos['U'].GetNbinsX()
    ny = histos['U'].GetNbinsY()
    xedges = [histos['U'].GetXaxis().GetBinLowEdge(i) for i in range(1, nx+2)]
    yedges = [histos['U'].GetYaxis().GetBinLowEdge(i) for i in range(1, ny+2)]
    resid_h2d = th2_from_matrix(resid, xedges, yedges, histos['U'].GetName() + "_U_pinvU_minus_I", "U*U^-1 - I")
    canvas2 = ROOT.TCanvas("c2", "c2", 800, 600)
    resid_h2d.Draw("colz")
    canvas2.SaveAs(os.path.join(outdir, "U_pinv_minus_I.png"))

    # Check dimensions!
    if  (pinv.shape[1] != R.shape[0]) or \
        (pinv.shape[0] != R.shape[0]) or \
        (U.shape[0] != R.shape[0]) or \
        (U.shape[1] != R.shape[0]) or \
        (ET.shape[0] != R.shape[0]) or \
        (T.shape[0] != R.shape[0]) or \
        (E.shape[0] != R.shape[0]):
        print(f"ERROR: Dimension mismatch!")
        sys.exit(1)
    
    # SVD check
    s = np.linalg.svd(U, compute_uv=False)
    plt.plot(s)
    plt.yscale('log')
    plt.xlabel('Index')
    plt.ylabel('Singular value')
    plt.title('Singular values of migration matrix U')
    plt.savefig(os.path.join(outdir, "singular_values_U.png"))
    plt.close()

    # Unfold 
    W = pinv @ R
    # compare to ET to check matrix inversion
    diff_WET = W - ET
    rel_diff_WET = np.divide(diff_WET, W, out=np.zeros_like(diff_WET), where=W!=0)
    print(f"Maximum absolute difference between unfolded W and ET: {np.max(np.abs(diff_WET))}")
    print(f"Maximum relative difference between unfolded W and ET: {np.max(np.abs(rel_diff_WET))}")
    print(f"chisquare of absolute difference between unfolded W and ET: {np.sum((diff_WET)**2)}/{len(ET)}")
    plot_difference(histos['ET'], mnvh2d_from_flattened(W, histos['ET']), os.path.join(outdir, "W_minus_ET_h.png"))  
    plt.figure(figsize=(10,6))
    plt.hist(diff_WET, alpha=0.7, label='Residuals (W - ET)', bins=100)
    plt.xlabel('Difference')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Residuals between Unfolded Histogram W and True Selected Histogram ET')
    plt.legend()
    plt.grid()
    # print on canvas number of rel_diff_WET != 0
    n_nonzero = np.sum(abs(rel_diff_WET) > 1.e-6)
    plt.text(0.7, 0.8, f'Non-zero residuals (>1e-6): {n_nonzero}/{len(rel_diff_WET)}', transform=plt.gca().transAxes)
    plt.savefig(os.path.join(outdir, "residuals_UinvR_minus_ET.png"))

    # test PURE refolding
    R_pure_ref = U @ W
    diff_Rref = R - R_pure_ref
    rel_diff_Rref = np.divide(diff_Rref, R, out=np.zeros_like(diff_Rref), where=R!=0)
    print(f"Maximum absolute difference between original R and pure refolded R_pure_ref: {np.max(np.abs(diff_Rref))}")
    print(f"RMS of relative difference between original R and pure refolded R_pure_ref: {np.sqrt(np.mean(rel_diff_Rref**2))}")
    print(f"chisquare of absolute difference between original R and pure refolded R_pure_ref: {np.sum((diff_Rref)**2)}/{len(R)}")  # add small number to avoid division by zero
    # plot the residuals    
    plt.figure(figsize=(10,6))
    plt.hist(rel_diff_Rref, alpha=0.7, label='Residuals (R - R_pure_ref)', bins=500)
    plt.xlabel('Difference')    
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('RELATIVE Residuals between Original Reco Histogram R and Pure Refolded Histogram R_pure_ref')
    plt.legend()
    plt.grid()
    # print on canvas number of rel_diff_Rref != 0
    n_nonzero = np.sum(abs(rel_diff_Rref) > 1.e-6   )
    plt.text(0.7, 0.8, f'Non-zero residuals (>1e-6): {n_nonzero}/{len(rel_diff_Rref)}', transform=plt.gca().transAxes)
    plt.savefig(os.path.join(outdir, "residuals_R_minus_Rpureref.png"))

    # And efficiency correct
    W = np.divide(W, E, out=np.zeros_like(W), where=E!=0)
    print("Unfolded histogram W computed.")

    # Compare to truth (T)
    diff_WT = W - T
    rel_diff_WT = np.divide(diff_WT, T, out=np.zeros_like(diff_WT), where=T!=0)
    print(f"Maximum absolute difference between unfolded W and truth T: {np.max(np.abs(diff_WT))}")
    print(f"Maximum relative difference between unfolded W and truth T: {np.max(np.abs(rel_diff_WT))}")
    print(f"chisquare of absolute difference between unfolded W and truth T: {np.sum((diff_WT)**2)}/{len(T)}")  # add small number to avoid division by zero
    # plot the residuals    
    plt.figure(figsize=(10,6))
    plt.hist(diff_WT, alpha=0.7, label='Residuals (W - T)', bins=500)
    plt.xlabel('Difference')    
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('Residuals between Unfolded Histogram W and Truth T')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(outdir, "residuals_W_minus_T.png"))

    # Re-folded histogram
    R_folded = U @ (np.multiply(W, E))
    print("Re-folded histogram R_folded computed.") 

    # Compare R and R_folded
    diff = R - R_folded
    rel_diff = np.divide(diff, R, out=np.zeros_like(diff), where=R!=0)
    print(f"Maximum absolute difference between original R and re-folded R_folded: {np.max(np.abs(diff))}")
    print(f"RMS of relative difference between original R and re-folded R_folded: {np.sqrt(np.mean(rel_diff**2))}")
    print(f"chisquare of absolute difference between original R and re-folded R_folded: {np.sum((diff)**2)}/{len(R)}")  # add small number to avoid division by zero
    # plot the residuals    
    plt.figure(figsize=(10,6))
    plt.hist(rel_diff, alpha=0.7, label='Residuals (R - R_folded)', bins=500)
    plt.xlabel('Difference')    
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.title('RELATIVE Residuals between Original Reco Histogram R and Re-folded Histogram R_folded')
    plt.legend()
    plt.grid()
    # print on canvas number of rel_diff != 0
    n_nonzero = np.sum(abs(rel_diff) > 1.e-6)
    plt.text(0.7, 0.8, f'Non-zero residuals (>1e-6): {n_nonzero}/{len(rel_diff)}', transform=plt.gca().transAxes)
    plt.savefig(os.path.join(outdir, "residuals_R_minus_Rfolded.png"))

    # ====================================================================
    # SAVE FINAL ROOT OUTPUTS
    # ====================================================================
    # Convert flattened arrays back to MnvH2D histograms for visualization or further analysis
    template = histos['R']
    R_h2d = mnvh2d_from_flattened(R, template)
    R_sel_h2d = mnvh2d_from_flattened(R_sel, template)
    ET_h2d = mnvh2d_from_flattened(ET, template)
    T_h2d = mnvh2d_from_flattened(T, template)

    R_folded_h2d = mnvh2d_from_flattened(R_folded, template)

    residual_h2d = R_h2d.Clone("R_minus_Rfolded")
    residual_h2d.Add(R_folded_h2d, -1)
    canvas_resid = ROOT.TCanvas("c_resid", "c_resid", 800, 600)
    residual_h2d.Draw("colz")
    residual_h2d.SetStats(0)
    canvas_resid.SaveAs(os.path.join(outdir, "R_minus_Rfolded.png"))


    



    # Save output histograms to a ROOT file
    output_file = os.path.join(outdir, "nulltest_output.root")
    # Detach histograms from any directory ownership to avoid double-delete
    for h in (R_h2d, R_sel_h2d, ET_h2d, T_h2d, R_folded_h2d):
        try:
            h.SetDirectory(0)
        except Exception:
            pass

    f_out = ROOT.TFile(output_file, "RECREATE")
    f_out.cd()
    R_h2d.Write("R")
    R_sel_h2d.Write("R_selection")
    ET_h2d.Write("ET_true_selected")
    T_h2d.Write("T_true")
    R_folded_h2d.Write("R_refolded")
    f_out.Close()
    print(f"Output histograms saved to {output_file}")

    # Close input ROOT files explicitly to avoid ROOT attempting to delete objects
    try:
        f_mig.Close()
    except Exception:
        pass
    try:
        f_sel.Close()
    except Exception:
        pass
    try:
        f_eff.Close()
    except Exception:
        pass

    # Remove Python references to large ROOT objects so they are not double-deleted
    del R_h2d, R_sel_h2d, ET_h2d, T_h2d, R_folded_h2d

if __name__ == '__main__':
    """Execute main pipeline when script is run directly."""
    main()
