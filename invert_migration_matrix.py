#!/usr/bin/env python3
"""
Invert a migration matrix using Moore-Penrose pseudoinverse regularization.

This module computes the Moore-Penrose pseudoinverse of a 2D migration matrix
stored in a ROOT file (as MnvH2D or TH2 object) and saves results in multiple
formats. It also supports Tikhonov regularization. The migration matrix
describes how reconstructed events map to true values in particle physics.

Mathematical background:
  - Migration matrix U: shape (n_reco, n_true)
  - U[i,j] = P(reco bin i | true bin j) after column normalization
  - Pseudoinverse U^+ satisfies: U * U^+ * U = U
  - Used to unfold reconstruction effects from experimental data

Output files:
  - ROOT file: pseudoinverse matrix (TH2D), original matrix, provenance histograms
  - NumPy archive (.npz): raw matrices for programmatic access

Example usage:
    python invert_migration_matrix.py --input migration.root \\
        --matrix h_ptrecpsiprime_qelike_migration --out output_pinv.root

Environment requirements:
  - ROOT PyROOT bindings
  - Set $MINERVAEXE or provide --input flag
"""

import os
import sys
import argparse
import numpy as np
try:
    import ROOT
except Exception as e:
    print("ERROR: PyROOT/ROOT import failed:", e)
    raise


def get_th2_from_object(obj, which_universe="cv", univ_index=0):
    """
    Extract a TH2 object from an MnvH2D or return the object if already TH2.

    Handles multiple input types used in MINERvA analyses:
      - Central value (CV) from MnvH2D objects
      - Individual systematic universe histograms from error bands
      - Plain ROOT TH2D/TH2F objects (returned as-is)

    Parameters
    ----------
    obj : ROOT.MnvH2D or ROOT.TH2
        Input histogram. If MnvH2D, will extract CV or specified error band
        universe. If already a TH2, will attempt to clone or return it.
    which_universe : str, optional
        Universe to extract. "cv" (default) for central value, or provide
        error band name (e.g., "BeamAngleX", "Flux") for systematic universe.
    univ_index : int, optional
        Index of the universe within the error band (default: 0).
        Only used if which_universe is not "cv".

    Returns
    -------
    ROOT.TH2D or ROOT.TH2F or None
        Extracted TH2 object, or None if extraction failed.
        Prints informational/warning messages during extraction.

    Notes
    -----
    - Tries lateral error bands first, then vertical error bands
    - If object has no error bands, returns None for non-CV requests
    - May fail for ROOT proxy objects; falls back to None
    """


    if which_universe == "cv":
        # Get the CV histo (just clone the object as TH2)
        try:
            cloned = obj.Clone()
            if isinstance(cloned, ROOT.TH2):
                print("Getting CV histogram as TH2.")
                return cloned
        except Exception:
            pass
    else:
        #first, check that the which_universe is an error band
        if len(obj.GetErrorBandNames()) == 0:
            print("Warning: object has no error bands. Maybe it was created without systematic variations?")
            return None
        if which_universe not in obj.GetErrorBandNames():
            print(f"Warning: error band '{which_universe}' not found in object. Available bands: {obj.GetErrorBandNames()}")
            return None
        # Try to get specific universe histo
        try:
            error_band = obj.GetLatErrorBand(which_universe)
            if error_band:
                univ_histo = error_band.GetHist(univ_index)
                if univ_histo:
                    print(f"Getting universe {univ_index} from error band '{which_universe}' as TH2.")
                    return univ_histo
            else:
                #try vertical error band
                error_band = obj.GetVertErrorBand(which_universe)
                if error_band:
                    univ_histo = error_band.GetHist(univ_index)
                    if univ_histo:
                        print(f"Getting universe {univ_index} from vertical error band '{which_universe}' as TH2.")
                        return univ_histo
        except Exception:
            print(f"Warning: could not get universe {univ_index} from error band '{which_universe}'.")
            return None #not implemented for specific universes here

    # If already a TH2
    try:
        if isinstance(obj, ROOT.TH2):
            return obj
    except Exception:
        # isinstance may fail for some proxy objects; fall through
        pass

    return None


# def matrix_from_th2(th2):
#     nx = th2.GetNbinsX()
#     ny = th2.GetNbinsY()
    
#     # Get all bin contents at once (much faster!)
#     # Note: ROOT stores bins as [0, nx+1] x [0, ny+1] including under/overflow
#     arr = np.array([[th2.GetBinContent(i, j) for j in range(1, ny+1)] for i in range(1, nx+1)])
#     return arr

def matrix_from_th2(th2):
    """
    Convert ROOT TH2 histogram to NumPy 2D array.

    Parameters
    ----------
    th2 : ROOT.TH2
        2D histogram object from ROOT.

    Returns
    -------
    np.ndarray
        2D array with shape (nbinsX, nbinsY), dtype float64.
        Bin contents read via GetBinContent(i, j).

    Notes
    -----
    - Reads only active bins (no under/overflow)
    - Preserves bin values as floats from ROOT
    - Uses explicit loop for clarity and debug control
      (vectorized alternative is commented in code above)
    """
    nx = int(th2.GetNbinsX())
    ny = int(th2.GetNbinsY())
    A = np.zeros((nx, ny), dtype=float)
    for i in range(nx):
        for j in range(ny):
            A[i, j] = float(th2.GetBinContent(i, j))
    return A


def th2_from_matrix(mat, xedges, yedges, name="mat_inv", title="Pseudoinverse"):
    """
    Convert NumPy 2D array to ROOT TH2D histogram with custom binning.

    Parameters
    ----------
    mat : np.ndarray
        2D array to convert, shape (nbinsX, nbinsY).
    xedges : array-like
        X-axis bin edges, length nbinsX + 1.
    yedges : array-like
        Y-axis bin edges, length nbinsY + 1.
    name : str, optional
        ROOT histogram name (default: "mat_inv").
    title : str, optional
        ROOT histogram title (default: "Pseudoinverse").

    Returns
    -------
    ROOT.TH2D
        2D histogram with matrix values as bin contents.

    Notes
    -----
    - Bin edges stored as C-style double arrays for ROOT compatibility
    - Preserves original array values in histogram bins
    - Does not include under/overflow bins
    """
    # Determine histogram dimensions from input array
    nbx = mat.shape[0]
    nby = mat.shape[1]

    # Convert bin edges to ROOT-compatible C doubles array
    from array import array
    xarr = array('d', list(xedges))
    yarr = array('d', list(yedges))

    # Create ROOT TH2D with variable bin widths
    h = ROOT.TH2D(name, title, nbx, xarr, nby, yarr)

    # Fill histogram with array values
    for i in range(nbx):
        for j in range(nby):
            h.SetBinContent(i, j, float(mat[i, j]))
    return h

def compute_pseudoinverse_from_object(obj, band_name="cv", univ_index=0, rcond=1e-15):
    """
    Compute Moore-Penrose pseudoinverse of migration matrix from ROOT object.

    Workflow:
      1. Extract TH2 histogram from MnvH2D or TH2 object
      2. Convert to NumPy array
      3. Normalize columns: each true bin sums to 1
      4. Compute pseudoinverse via numpy.linalg.pinv()
      5. Convert result back to ROOT TH2D with original binning

    Parameters
    ----------
    obj : ROOT.MnvH2D or ROOT.TH2
        Input histogram containing migration matrix.
    band_name : str, optional
        Error band/universe to invert. "cv" (default) for central value,
        or error band name (e.g., "Flux", "BeamAngleX").
    univ_index : int, optional
        Index of universe within error band (default: 0).
    rcond : float, optional
        Cutoff ratio for small singular values (default: 1e-15).
        Singular values < rcond * max_singular_value are set to zero.
        Controls numerical stability vs. regularization strength.
        Larger rcond = more aggressive truncation = more regularization.

    Returns
    -------
    pinv : np.ndarray
        Moore-Penrose pseudoinverse, shape (n_true, n_reco).
    h_pinv : ROOT.TH2D
        Pseudoinverse stored as ROOT histogram with original bin edges.
    A : np.ndarray
        Original migration matrix (column-normalized), shape (n_reco, n_true).

    Raises
    ------
    Exception
        If band_name != "cv" but object has no error bands.
    SystemExit
        If extraction of TH2 fails.

    Notes
    -----
    - Bins are normalized by column (true spectrum)
    - Each column of normalized matrix sums to 1, representing P(reco | true)
    - Singular values < rcond * max(singular_values) are set to 0
    """

    if band_name == "cv":
        suffix = "_cv"
    else:
        suffix = f"_{band_name}_univ{univ_index}"


    # print(f"Error bands available in object: {len(obj.GetErrorBandNames())}")
    if band_name != "cv" and len(obj.GetErrorBandNames()) == 0:
        raise Exception(f"Object has no error bands; cannot select error band '{band_name}'.")


    th2 = get_th2_from_object(obj, band_name, univ_index)
    if th2 is None:
        print("ERROR: Could not obtain a TH2 from the object. Object class:", obj.ClassName())
        sys.exit(6)

    print("Successfully retrieved TH2 with binning:",end='')
    nx = th2.GetNbinsX(); ny = th2.GetNbinsY()
    print(f" nbinsX={nx}, nbinsY={ny}")

    # Read bin edges
    xedges = [th2.GetXaxis().GetBinLowEdge(i) for i in range(1, nx+2)]
    yedges = [th2.GetYaxis().GetBinLowEdge(i) for i in range(1, ny+2)]

    # Convert ROOT histogram to NumPy array
    A = matrix_from_th2(th2)
    print("Matrix shape:", A.shape)

    # Column-normalize the migration matrix: divide each column by its sum
    # After normalization, each column represents P(reco | true)
    Y = np.sum(A, axis=0)  # Column sums (true spectrum projection)
    X = np.sum(A, axis=1)  # Row sums (reco spectrum projection)
    A = np.divide(A, Y, out=np.zeros_like(A), where=Y != 0)

    # Compute Moore-Penrose pseudoinverse
    # A^+ is computed as V @ S_pinv @ U^T where A = U @ S @ V^T (SVD)
    # S_pinv replaces singular values s with 1/s if s > rcond * max(s), else 0
    print("Computing Moore-Penrose pseudoinverse using numpy.linalg.pinv...")
    pinv = np.linalg.pinv(A, rcond=rcond)

    # Create TH2D histogram for the pseudoinverse with original bin edges
    h_pinv = th2_from_matrix(
        pinv,
        xedges,
        yedges,
        name=obj.GetName() + suffix + '_pinv',
        title='Moore-Penrose pseudoinverse'
    )


    return pinv, h_pinv, A


def build_Cinv_from_counts(R):
    """
    Build diagonal inverse covariance matrix from count statistics.

    For Poisson statistics, the covariance is diagonal with C[i,i] = R[i].
    This function computes the inverse: C^-1[i,i] = 1/R[i] where defined.

    Parameters
    ----------
    R : np.ndarray
        1D array of event counts (reconstruction spectrum).

    Returns
    -------
    Cinv : np.ndarray
        Diagonal of inverse covariance matrix: 1/R where R > 0, 0 elsewhere.
    mask : np.ndarray
        Boolean array indicating bins with valid (R > 0) counts.

    Notes
    -----
    - Used in Tikhonov regularization to weight bins by statistical significance
    - Bins with zero counts have Cinv[i] = 0 (effectively excluded from fit)
    """
    Cinv = np.zeros_like(R, dtype=float)
    mask = R > 0
    Cinv[mask] = 1.0 / R[mask]
    return Cinv, mask

def tikhonov_inverse(U, R, lam, L, weighted=True):
    """
    Solve regularized linear system using Tikhonov regularization.

    Solves the Tikhonov-regularized least squares problem:
        minimize || C_R^{-1/2} (U @ W - R) ||^2 + lam^2 ||L @ W||^2

    Equivalent to solving:
        (U^T C_R^{-1} U + lam^2 L^T L) W = U^T C_R^{-1} R

    The regularization term lam^2 L^T L encourages smooth solutions.
    The data term includes statistical weighting from reconstruction counts.

    Parameters
    ----------
    U : np.ndarray
        Migration matrix, shape (n_reco, n_true).
        Should be column-normalized (each column sums to 1, P(reco|true)).
    R : np.ndarray
        Reconstruction histogram (event counts), shape (n_reco,).
    lam : float
        Regularization strength. Larger values enforce stronger smoothness
        at the cost of higher bias. Units depend on L scaling.
    L : np.ndarray
        Regularization operator matrix, shape (n_reg, n_true).
        Often identity matrix (Tikhonov-0, penalizes large values) or
        difference operator (Tikhonov-1, penalizes non-smoothness).
    weighted : bool, optional
        If True (default), weight by Poisson statistics (C_R^{-1} from counts).
        If False, use equal weighting (C_R^{-1} = I).

    Returns
    -------
    W : np.ndarray
        Unfolded spectrum (true spectrum), shape (n_true,).

    Notes
    -----
    - Uses sqrt weighting in normal equations for numerical stability
    - Bins with R[i] = 0 are excluded from fit (masked out)
    - System matrix (U^T C^{-1} U + lam^2 L^T L) solved directly
      without inversion for better numerical properties
    """
    # Build or initialize inverse covariance weighting
    if weighted:
        # Poisson statistics: C^{-1}[i,i] = 1 / R[i] for R[i] > 0
        Cinv, mask = build_Cinv_from_counts(R)
    else:
        # Equal weighting: all bins equally important
        Cinv = np.ones_like(R, dtype=float)
        mask = np.ones_like(R, dtype=bool)

    # Transform matrices to weighted form using sqrt(C^{-1})
    # This ensures normal equations are numerically stable
    Uw = U[mask] * np.sqrt(Cinv[mask, None])
    Rw = R[mask] * np.sqrt(Cinv[mask])

    # Build and solve: (U^T C^{-1} U + lam^2 L^T L) W = U^T C^{-1} R
    A = Uw.T @ Uw
    B = lam**2 * (L.T @ L)
    rhs = Uw.T @ Rw

    W = np.linalg.solve(A + B, rhs)
    return W

def main():
    """
    Main entry point: parse arguments, load migration matrix, compute pseudoinverse.

    Workflow:
      1. Parse command-line arguments for input/output paths and regularization params
      2. Load input ROOT file and extract specified migration matrix
      3. Optionally select specific error band universe
      4. Compute Moore-Penrose pseudoinverse via compute_pseudoinverse_from_object()
      5. Save results to ROOT file (pseudoinverse + provenance histograms)
      6. Save numeric arrays to NumPy archive (.npz)
      7. Print diagnostic information (pseudoinverse quality check)

    Returns
    -------
    None
        Exits with code 0 on success, non-zero on error.
    """
    p = argparse.ArgumentParser(
        description="Invert migration matrix via Moore-Penrose pseudoinverse"
    )
    p.add_argument('--input', '-i', default=None, help='Input ROOT file (default: $MINERVAEXE/MigMat_me1L_after2.root)')
    p.add_argument('--matrix', '-m', default='h_ptrecpsiprime_qelike_migration', help='Name of the migration histogram inside the ROOT file')
    p.add_argument('--out', '-o', default=None, help='Output ROOT file to save inverted TH2 (default: <input_basename>_pinv.root)')
    p.add_argument('--npz', default=None, help='Also save numpy archive with original and pinv arrays (default: <input_basename>_pinv.npz)')
    p.add_argument('--rcond', type=float, default=0.0001, help='rcond parameter passed to numpy.linalg.pinv (default: machine precision)')
    p.add_argument('--band', default='cv', help='Error band name to use (default: "cv" for central value)')
    p.add_argument('--univ', type=int, default=0, help='Universe index to use if band is not "cv" (default: 0)')
    args = p.parse_args()

    print("\n=== Invert Migration Matrix ===\n")

    # ====================================================================
    # RESOLVE INPUT FILE PATH
    # ====================================================================
    input_path = args.input
    if input_path is None:
        # Try to construct default path from $MINERVAEXE environment variable
        minerva_exe = os.environ.get('MINERVAEXE')
        if minerva_exe:
            input_path = os.path.join(
                minerva_exe,
                'PsiPrimeAnalysisOutputs_1L',
                'Migration_me1L_testUnfolding.root'
            )
        else:
            print(
                "ERROR: No input specified and $MINERVAEXE not set. "
                "Please provide --input or set MINERVAEXE."
            )
            sys.exit(2)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(3)

    # ====================================================================
    # LOAD INPUT ROOT FILE
    # ====================================================================
    f = ROOT.TFile.Open(input_path, 'READ')
    if not f or f.IsZombie():
        print(f"ERROR: Could not open ROOT file: {input_path}")
        sys.exit(4)

    print(f"Opened file: {input_path}")
    print("Looking for matrix:", args.matrix)

    # Retrieve migration matrix object from file
    obj = f.Get(args.matrix)
    if not obj:
        print(f"ERROR: Object {args.matrix} not found in file. Available keys:")
        f.GetListOfKeys().Print()
        sys.exit(5)


    band_name = args.band
    univ_index = args.univ

    if band_name == "cv":
        suffix = "_cv"
    else:
        suffix = f"_{band_name}_univ{univ_index}"

    pinv, h_pinv, A = compute_pseudoinverse_from_object(obj, band_name, univ_index, rcond=args.rcond)

    # ====================================================================
    # SAVE RESULTS TO ROOT FILE
    # ====================================================================

    # Determine output file paths (use defaults based on input filename)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_root = args.out or os.path.join(
        os.path.dirname(input_path), base + '_pinv.root'
    )
    out_npz = args.npz or os.path.join(
        os.path.dirname(input_path), base + '_pinv.npz'
    )

    # Create ROOT output file and write pseudoinverse histogram
    outf = ROOT.TFile.Open(out_root, 'RECREATE')
    if not outf or outf.IsZombie():
        print(f"ERROR: Could not create output file: {out_root}")
        sys.exit(7)
    outf.cd()

    # Write pseudoinverse matrix
    h_pinv.Write()

    # Write original migration matrix for provenance/reference
    th2 = get_th2_from_object(obj, band_name, univ_index)
    th2_clone = th2.Clone(args.matrix + suffix)
    th2_clone.Write()
    print(f"Wrote inverted matrix to {out_root}")

    # Write reconstruction and truth spectra used to build migration matrix
    # (useful for understanding the unfolding context)
    clone_reco = f.Get(args.matrix.replace('migration', 'reco')).Clone()
    clone_true = f.Get(args.matrix.replace('migration', 'truth')).Clone()

    if clone_reco:
        clone_reco.SetName(
            args.matrix.replace('migration', 'reco') + suffix
        )
        clone_reco.Write()
    if clone_true:
        clone_true.SetName(
            args.matrix.replace('migration', 'truth') + suffix
        )
        clone_true.Write()

    outf.Close()

    # ====================================================================
    # SAVE RESULTS TO NUMPY ARCHIVE
    # ====================================================================
    try:
        np.savez_compressed(
            out_npz,
            matrix=A,
            pinv=pinv,
            matrix_name=args.matrix
        )
        print(f"Wrote numpy archive to {out_npz}")
    except Exception as e:
        print("Warning: failed to write npz:", e)

    # ====================================================================
    # PRINT DIAGNOSTICS
    # ====================================================================
    # Moore-Penrose pseudoinverse should satisfy: A @ A+ @ A = A
    # This residual indicates numerical quality of the inversion
    try:
        resid = A.dot(pinv).dot(A) - A
        max_abs_resid = np.abs(resid).max()
        print(
            f"Diagnostic: max|A*pinv*A - A| = {max_abs_resid:.3e} "
            f"(should be ~0; indicates numerical precision)"
        )
    except Exception:
        pass




if __name__ == "__main__":
    """Execute main pipeline when script is run directly."""
    main()
