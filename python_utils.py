def plot_psiprime_systematics(
    histo_to_draw,
    error_band_names="all",
    output_folder="universes_plots",
    projection_axis=None, # by default, expect a 1D histogram
    name=None,
    plot_style=None
):
    '''
    Analyze and plot Psi' peak systematics for a given histogram.
    Args:
        histo_to_draw: MnvH1D or MnvH2D histogram (with systematics)
        error_band_names: "all", "vert", "lat", or list of error band names
        output_folder: where to save plots and text
        projection_axis: 0=X, 1=Y (for MnvH2D)
        name: optional, for output naming
        plot_style: optional, for future style customization
    '''
    import ROOT, os, matplotlib.pyplot as plt, numpy as np
    ROOT.TH1.AddDirectory(False)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Project if needed
    if projection_axis is not None:
        if hasattr(histo_to_draw, "ProjectionX") and hasattr(histo_to_draw, "ProjectionY"):
            if projection_axis==0:
                histo_to_draw = histo_to_draw.ProjectionX()
            else:
                histo_to_draw = histo_to_draw.ProjectionY()
    # if projection_axis is None: expect a 1D histogram
    if name is None:
        name = histo_to_draw.GetName()
    # Error band names
    error_band_names_title = error_band_names
    if isinstance(error_band_names, str):
        if error_band_names == "all":
            lat_error_band_names = histo_to_draw.GetLatErrorBandNames()
            vert_error_band_names = histo_to_draw.GetVertErrorBandNames()
            error_band_names = list(lat_error_band_names) + list(vert_error_band_names)
        elif error_band_names == "vert":
            error_band_names = list(histo_to_draw.GetVertErrorBandNames())
        elif error_band_names == "lat":
            error_band_names = list(histo_to_draw.GetLatErrorBandNames())
        else:
            error_band_names = [error_band_names]
    print(f"\nError bands: {error_band_names}\n\n")
    # Collect universes
    all_universes = []
    h_test = ROOT.TH1D("h_test","h_test",100,-2,2)
    for error_band_name in error_band_names:
        vertical_bool = False
        lateral_bool = False
        error_band = histo_to_draw.GetVertErrorBand(error_band_name)
        if error_band:
            vertical_bool = True
        else:
            error_band = histo_to_draw.GetLatErrorBand(error_band_name)
            if error_band:
                lateral_bool = True
        if not vertical_bool and not lateral_bool:
            print(f"Error: Error band {error_band_name} not found for this histogram.")
            continue
        nUniverses = error_band.GetNHists()
        print(f"Processing {error_band_name}: {nUniverses} universes")
        this_error_band_universes = []
        for i in range(nUniverses):
            h_thisUniverse = error_band.GetHist(i)
            if not h_thisUniverse:
                print(f"Error: Universe {i} not found in the error band {error_band_name}.")
                continue
            mean = h_thisUniverse.GetMean()
            this_error_band_universes.append( (i, h_thisUniverse) )
            h_test.Fill(mean)
        all_universes.append((error_band_name,this_error_band_universes))
    # c_test = ROOT.TCanvas("c_test","c_test",800,600)
    # # no line, fill with grey
    # histo_to_draw.SetFillColor(ROOT.kGray)
    # histo_to_draw.Draw("hist")
    # c_test.Update()
    print(f"\n {len(all_universes)} error bands")
    for sublist in all_universes:
        print(f"  Error band {sublist[0]}: {len(sublist[1])} universes")
    total_average = 0
    total_rms = 0
    syst_uncertainties = []
    def lateral_or_vertical(error_band_name, histo):
        error_band = histo.GetVertErrorBand(error_band_name)
        if error_band:
            return "vertical"
        else:
            error_band = histo.GetLatErrorBand(error_band_name)
            if error_band:
                return "lateral"
        return "none"
    for sublist in all_universes:
        psiprime_peaks = []
        psiprime_peak_errors = []
        error_band_name = sublist[0]
        universes = sublist[1]
        print(f"Processing {len(universes)} universes for error band {error_band_name}")
        c2 = ROOT.TCanvas(f"c{error_band_name}",f"c{error_band_name}",800,600)
        c2.cd()
        histo_to_draw_clone = histo_to_draw.Clone()
        histo_to_draw_clone.Scale(1.0,"width")
        histo_to_draw_clone.SetLineColor(0)
        histo_to_draw_clone.SetFillColor(ROOT.kGray)
        histo_to_draw_clone.Draw("hist")
        for universe_index, h in universes:
            mean = h.GetMean()
            title_sanitized = str(error_band_name).replace(" ","_").replace("/","_").replace("-","_")
            h.SetTitle(f"{name} {error_band_name} universe {universe_index}; #psi'; Entries")
            h.SetTitleFont(42)
            h.SetTitleSize(0.05,"")
            h.Scale(1.0,"width")
            if h.GetEntries()<1:
                print(f"Skipping {title_sanitized} universe {universe_index}: empty histogram.")
                continue
            c2.cd()
            if universe_index==0:
                h.Draw("hist same")
            else:
                h.Draw("hist same")
            c2.Update()
            max_bin_pos = h.GetBinCenter(h.GetMaximumBin())
            sigma = h.GetRMS()
            xmin = max(-4, max_bin_pos - 1*sigma)
            xmax = min(10, max_bin_pos + 1*sigma)
            fitResult = ROOT.TFitResultPtr( h.Fit("gaus", "SQL", "", xmin, xmax) )
            if not fitResult or not hasattr(fitResult, "Status"):
                print(f"Fit returns null ptr for {title_sanitized} universe {universe_index}")
                continue
            status = fitResult.Status()
            if status != 0:
                print(f"Fit failed for {title_sanitized} universe {universe_index} with status {status}")
                continue
            fitFunc = h.GetFunction("gaus")
            if fitFunc is not None:
                fitFunc.SetLineColor(h.GetLineColor())
                fitFunc.SetLineStyle(h.GetLineStyle())
                fitFunc.Draw("same")
                c2.Update()
                fit_mean = fitFunc.GetParameter(1)
                fit_mean_error = fitFunc.GetParError(1)
                psiprime_peaks.append(fit_mean)
                psiprime_peak_errors.append(fit_mean_error)
                print(f"peak: {fit_mean:.6}")
        tlatex = ROOT.TLatex()
        tlatex.SetNDC()
        tlatex.SetTextFont(42)
        tlatex.SetTextSize(0.04)
        tlatex.SetTextAlign(22)
        tlatex.DrawLatex(0.5,0.95,f"Error band {error_band_name} ({len(psiprime_peaks)} universes)")
        c2.SaveAs(os.path.join(output_folder,f"{name}_{error_band_name}.png"))
        h_peaks = ROOT.TH1D("h_peaks","h_peaks",100,-1,1)
        for peak in psiprime_peaks:
            h_peaks.Fill(peak)
        plt.figure(figsize=(10,7))
        plt.hist(psiprime_peaks, alpha=0.7, color='blue')
        # add vertical line with mean value
        peaks_mean = np.mean(psiprime_peaks)
        plt.axvline(peaks_mean, color='red', linestyle='dashed', linewidth=3)
        # add vertical line with central value from histo_to_draw
        max_bin_pos = histo_to_draw.GetBinCenter(histo_to_draw.GetMaximumBin())
        sigma = histo_to_draw.GetRMS()
        xmin = max(-4, max_bin_pos - 1*sigma)
        xmax = min(10, max_bin_pos + 1*sigma)
        fitResult = ROOT.TFitResultPtr( histo_to_draw.Fit("gaus", "SQL", "", xmin, xmax) )
        if not fitResult or not hasattr(fitResult, "Status"):
            print(f"Fit returns null ptr for {title_sanitized} universe {universe_index}")
            continue
        status = fitResult.Status()
        if status != 0:
            print(f"Fit failed for {title_sanitized} universe {universe_index} with status {status}")
            continue
        fitFunc = histo_to_draw.GetFunction("gaus")
        if fitFunc is not None:
            fitFunc.SetLineColor(histo_to_draw.GetLineColor())
            fitFunc.SetLineStyle(histo_to_draw.GetLineStyle())
            fitFunc.Draw("same")
            c2.Update()
            fit_mean = fitFunc.GetParameter(1)
            fit_mean_error = fitFunc.GetParError(1)
        plt.axvline(fit_mean, color='green', linestyle='dashed', linewidth=3)

        plt.title(f"{name} {error_band_name} Psi' peak positions", fontsize=16)
        plt.xlabel(r"$\psi'$")
        plt.ylabel("Entries")
        peaks_sigma = 0
        for peak in psiprime_peaks:
            peaks_sigma += (peak - peaks_mean)**2
        peaks_sigma = np.sqrt(peaks_sigma / (len(psiprime_peaks)))
        universes = len(psiprime_peaks)
        plt.text(0.6, 0.8, f"Mean: {peaks_mean:.3f}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.6, 0.75, f"RMS: {peaks_sigma:.3f}", transform=plt.gca().transAxes, fontsize=12)
        plt.text(0.6, 0.7, f"Universes: {universes}", transform=plt.gca().transAxes, fontsize=12)
        plt.grid()
        plt.savefig(os.path.join(output_folder,f"{name}_{error_band_name}_peaks.png"))
        plt.close()
        total_rms += peaks_sigma**2
        total_average += peaks_mean
        syst_uncertainties.append( (error_band_name, peaks_sigma) )
    total_average = total_average / len(all_universes)
    total_rms = np.sqrt(total_rms)
    print(f"\npsi' = {total_average:.3f} +- {total_rms:.3f} (syst)")
    syst_uncertainties.sort(key=lambda x: x[1], reverse=True)
    print("\nSystematic uncertainties by error band:")
    lat_or_vert = []
    for band, uncertainty in syst_uncertainties:
        lat_or_vert.append( lateral_or_vertical(band,histo_to_draw) )
    with open(os.path.join(output_folder,"syst_uncertainties.txt"),"w") as f:
        f.write("Systematic uncertainties by error band:\n")
    for i, (band, uncertainty) in enumerate(syst_uncertainties):
        print(f"  {band}: {uncertainty:.3f} ({lat_or_vert[i]})")
        with open(os.path.join(output_folder,"syst_uncertainties.txt"),"a") as f:
            f.write(f"  {band}: {uncertainty:.3f} ({lat_or_vert[i]})\n")
    with open(os.path.join(output_folder,"syst_uncertainties.txt"),"a") as f:
        f.write(f"\nTotal systematic uncertainty on the psi' peak position: {total_rms:.3f}\n")
    print(f"Systematic uncertainties written to {os.path.join(output_folder,'syst_uncertainties.txt')}")
import ROOT
ROOT.gROOT.SetBatch(1)
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from typing import Optional, List, Dict, Tuple
from array import array
import matplotlib as mpl



from PlotUtils import Plot
print("PlotUtils imported successfully")

# Need to load MnvH2D and MnvH1D classes
from PlotUtils import LoadPlotUtilsLib
print("LoadPlotUtilsLib imported successfully")

input_file = None

# keys = input_file.GetListOfKeys()
# objects = [key.GetName() for key in keys]
# # get object types
# object_types = [key.GetClassName() for key in keys]
# with open('histogram_names.txt', 'w') as f:
#     for obj, obj_type in zip(objects, object_types):
#         f.write(f"{obj_type}\t|\t({obj})\n")
# print("Histogram names written to histogram_names.txt")


###################################################################################################################################

def get_histo_from_file(filename, histogram_name):
    # Open input file
    input_file = ROOT.TFile.Open(filename)
    if not input_file or input_file.IsZombie():
        print("Error opening file:", filename)
        return None
    print("Input file opened successfully\n\n")
    # Get the histogram
    h_mc = input_file.Get(histogram_name)
    if not h_mc:
        print(f"Error: Histogram {histogram_name} not found in the file.")
        return None

    return h_mc

##################################################################################################################################

def getName(string):
    """
    Returns what's betweeen the last '/' and the last '.' in a string.
    """
    return string.split("/")[-1].split(".")[0]

#################################################################################################################################
def find_objects_with_name(name):
    keys = input_file.GetListOfKeys()
    objects = [key.GetName() for key in keys]
    objects = [obj for obj in objects if name in obj]
    print(f"Objects containing '{name}':")
    for obj in objects:
        print(obj)
#################################################################################################################################

#################################################################################################################################
def get_class_name(obj):
    """
    Returns the class name of the object.
    """
    return obj.__class__.__name__
#################################################################################################################################

#################################################################################################################################
def plot_and_save_simple(name, draw_option="",logy=False, divide_by_bin_width=True, debug_check_bins=False, titlex="", titley="",title="",prefix="",out_dir=".", logz=False):
    if type(name) == str:
        h = input_file.Get(name)
    elif get_class_name(name) == "MnvH2D":
        h = name
        name = name.GetName()
    elif get_class_name(name) == "MnvH1D":
        h = name
        name = name.GetName()
    else:
        print("plot_and_save_simple: first argument is invalid!!")

    if not h:
        print(f"Histogram {name} not found in the file.")
        return
    if divide_by_bin_width:
        h.Scale(1, "width")
    


    # check bins
    if(debug_check_bins):
        if(h.GetDimension() == 2):
            # 2D histogram
            print(f"2D histogram {name} found")
            # get x axis bin edges
            x_bins = h.GetXaxis().GetNbins()
            x_bin_edges = [h.GetXaxis().GetBinLowEdge(i) for i in range(1, x_bins + 1)]
            x_bin_hedges = [h.GetXaxis().GetBinUpEdge(i) for i in range(1, x_bins + 1)]
            print(f"x bin edges({x_bins}): {x_bin_edges}")
            print(f"x bin hedges({x_bins}): {x_bin_hedges}")
            # get y axis bin edges
            y_bins = h.GetYaxis().GetNbins()
            y_bin_edges = [h.GetYaxis().GetBinLowEdge(i) for i in range(1, y_bins + 1)]
            y_bin_hedges = [h.GetYaxis().GetBinUpEdge(i) for i in range(1, y_bins + 1)]
            print(f"y bin edges({y_bins}): {y_bin_edges}")
            print(f"y bin hedges({y_bins}): {y_bin_hedges}")
    
    
    # plot (using pyROOT)
    c = ROOT.TCanvas("c", "Canvas", 800, 600)
    c.SetRightMargin(0.15)
    h.Draw(draw_option)
    h.GetXaxis().SetTitle(titlex)
    h.GetYaxis().SetTitle(titley)
    h.SetTitle(title)


    if logy:
        c.SetLogy()
    if logz:
        c.SetLogz()
    c.SaveAs(f"{out_dir}/{prefix}{name}.png")


#################################################################################################################################

# Color-blind friendly palette (Okabe–Ito). Use hex to ensure consistent colors across ROOT versions.
def _cb(hex_str: str) -> int:
    return ROOT.TColor.GetColor(hex_str)

colors = {
    # Base categories
    "qe": _cb("#0072B2"),         # Blue
    "2p2h": _cb("#E69F00"),       # Orange
    "res": _cb("#009E73"),        # Bluish green
    "dis": _cb("#D55E00"),        # Vermillion
    "oth": _cb("#CC79A7"),        # Reddish purple

    # QE-like breakdown (match base categories where applicable)
    "qelike_qe": _cb("#0072B2"),
    "qelike_2p2h": _cb("#E69F00"),
    "qelike_res": _cb("#009E73"),
    "qelike_dis": _cb("#D55E00"),

    # Optional complementary ("not" categories) – lighter/alternate but still color-blind safe
    "qelikenot_qe": _cb("#56B4E9"),   # Sky blue
    "qelikenot_2p2h": _cb("#F0E442"), # Yellow
    "qelikenot_res": _cb("#CC79A7"),  # Reddish purple
    "qelikenot_dis": _cb("#999999"),  # Grey
}

#################################################################################################################################
def plot_mc_breakdown_stack(
    name: str,
    draw_option: str = "hist",
    logy: bool = False,
    divide_by_bin_width: bool = True,
    normalize_to_unity: bool = False,
    legend_pos: Tuple[float, float, float, float] = (0.70, 0.65, 0.90, 0.90),
    title: Optional[str] = None,
    titlex: Optional[str] = None,
    titley: Optional[str] = None,
    out_dir: str = ".",
    filename_prefix: str = "",
    components: Optional[List[str]] = None,
    colors_override: Optional[Dict[str, int]] = None,
    y_min: Optional[float] = 0.0,
    y_max: Optional[float] = None,
    save_formats: Tuple[str, ...] = ("png",),
    pot_scale: float = 1.0,
):
    """
    Plot and save a stacked breakdown of MC components for a given histogram prefix.

    Expected histograms in the ROOT file (with 'name' as prefix):
      name_qelike_qe, name_qelike_2p2h, name_qelike_res, name_qelike_dis, name_oth

    QE is drawn FIRST (bottom of the stack).

    Options:
    - divide_by_bin_width: scale each hist by bin width
    - normalize_to_unity: normalize each component to unit area (after optional width scaling)
    - legend_pos: (x1,y1,x2,y2)
    - colors_override: mapping component->ROOT color overrides
    - y_min/y_max: y-axis range overrides
    - save_formats: tuple of extensions to save (e.g., ("png","pdf"))
    """

    # Default components with QE first
    breakdown_list = components if components is not None else [
        "qelike_qe",
        "qelike_2p2h",
        "qelike_res",
        "qelike_dis",
        "oth",
    ]

    # Human-friendly labels
    label_map = {
        "qelike_qe": "QE",
        "qelike_2p2h": "2p2h",
        "qelike_res": "RES",
        "qelike_dis": "DIS",
        "oth": "Other",
    }

    # Colors
    colormap = dict(colors)
    if colors_override:
        colormap.update(colors_override)

    # Load histograms (clone to avoid altering originals)
    hists = []
    ref_hist = None
    for comp in breakdown_list:
        h = input_file.Get(f"{name}_{comp}")
        if not h:
            print(f"[plot_mc_breakdown_stack] Warning: missing histogram {name}_{comp}; skipping.")
            continue
        h_clone = h.Clone(f"{name}_{comp}_clone")
        h_clone.SetDirectory(0)
        if divide_by_bin_width:
            h_clone.Scale(1, "width")
        if normalize_to_unity:
            integral = h_clone.Integral()
            if integral > 0:
                h_clone.Scale(1.0 / integral)
        elif pot_scale != 1.0:
            h_clone.Scale(pot_scale)
        # Style
        # h_clone.SetFillColor(colormap.get(comp, ROOT.kGray+1))
        h_clone.SetLineColor(colormap.get(comp, ROOT.kBlack))
        h_clone.SetFillColorAlpha(colormap.get(comp, ROOT.kGray+1), 0.7)
        h_clone.SetLineWidth(2)
        hists.append((comp, h_clone))
        if ref_hist is None:
            ref_hist = h_clone

    if not hists:
        print(f"[plot_mc_breakdown_stack] Nothing to draw for prefix '{name}'.")
        return None

    # Build stack (order preserved; first goes to bottom)
    stack = ROOT.THStack(f"{name}_stack", "")
    for comp, h in hists:
        stack.Add(ROOT.TH1D(h))


    # Canvas
    os.makedirs(out_dir, exist_ok=True)
    c = ROOT.TCanvas(f"c_{name}_stack", "Canvas", 900, 700)

    # Draw stack first to compute max
    stack.Draw(draw_option)

    # Titles and axes
    # X axis titles
    final_title = title if title is not None else f"{name} QE-like MC Breakdown"
    stack.SetTitle(final_title)
    if "psiprime" in name:
        x_var = "#psi\'"
        stack.GetXaxis().SetTitle(x_var)
    if "EnuTrueMinusErec" in name:
        x_var = "(E_{#nu}^{RE} - E_{#nu}^{true}) / E_{#nu}^{true}"
        stack.GetXaxis().SetTitle(x_var)
    # Y axis titles
    if ref_hist is not None:
        x_title = titlex if titlex is not None else ref_hist.GetXaxis().GetTitle()
        bin_width = ref_hist.GetBinWidth(1) if ref_hist.GetNbinsX() > 0 else 1.0
        y_title_default = f"dN/d{x_var} " if divide_by_bin_width else f"Events / {bin_width}"
        y_title = titley if titley is not None else y_title_default
        stack.GetYaxis().SetTitle(y_title)

    # Y range
    max_val = stack.GetMaximum()
    if y_max is None:
        y_max = max_val * (2.0 if logy else 1.25)
    if y_min is None:
        y_min = 1e-3 if logy else 0.0
    stack.SetMaximum(y_max)
    stack.SetMinimum(y_min)

    # Legend
    leg = ROOT.TLegend(*legend_pos)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    for comp, h in hists:
        leg.AddEntry(h, label_map.get(comp, comp), "f")
    leg.Draw()

    # pad and titles formatting. 

    stack.GetXaxis().SetTitleSize(0.06)
    stack.GetYaxis().SetTitleSize(0.06)
    stack.GetYaxis().SetTitleOffset(1.5)
    stack.GetXaxis().SetTitleFont(42)
    stack.GetYaxis().SetTitleFont(42)
    stack.GetXaxis().SetLabelSize(0.05)
    stack.GetYaxis().SetLabelSize(0.05)


    c.Modified()
    c.Update()
    c.SetLeftMargin(0.2)
    c.SetBottomMargin(0.15)
    c.Modified()
    c.Update()

    # Grid, log
    c.SetGrid()
    if logy:
        c.SetLogy()

    # Save
    base_name = f"{filename_prefix}{name}_breakdown_stack"
    for ext in save_formats:
        c.SaveAs(os.path.join(out_dir, f"{base_name}.{ext}"))

    return c

#################################################################################################################################
def plot_mc_data_shapeOnly(name, draw_option="", logy=False, do_breakdown=True,out_dir="."):

    h_mc = input_file.Get(name + "_mc")
    if not h_mc:
        print(f"Histogram {name}_mc not found in the file.")
        return

    if do_breakdown:
        breakdown_list=["qelike_qe","qelike_2p2h","qelike_res","qelike_dis","oth"]
        #open all breakdown histograms
        h_mc_breakdown = [input_file.Get(name + "_" + breakdown) for breakdown in breakdown_list]
        # creat a stack
        h_mc_stack = ROOT.THStack("h_mc_stack", "MC Breakdown")
        # create a legend
        legend_stack = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        # loop through the breakdown histograms
        for i, h in enumerate(h_mc_breakdown):
            if not h:
                print(f"Histogram {name + breakdown_list[i]} not found in the file.")
                return
            h.Scale(1, "width")
            # set the color and style
            h.SetFillColor(colors[breakdown_list[i]])
            # add the histograms to the stack
            h_mc_stack.Add(ROOT.TH1D(h))
            legend_stack.AddEntry(h, breakdown_list[i], "f")
    #data
    h_data = input_file.Get(name + "_data")
    if not h_data:
        print(f"Histogram {name}_data not found in the file.")
        # return
    # this function is only for 1D histos
    if h_mc.GetDimension() != 1 or h_data.GetDimension() != 1:
        print(f"Histogram {name} is 2D, this function only works for 1D histograms.")
        return
    
    # get pot scale
    pot = input_file.Get("pot")
    #potscale data/mc
    print(f"POT data: {pot.X()}, POT mc: {pot.Y()}")
    potscale = pot.X()/pot.Y()

    # rescale number of events
    h_mc.Scale(potscale)
    if (do_breakdown):
        # rescale breakdown histograms
        for i, h in enumerate(h_mc_breakdown):
            h.Scale(potscale)
    # divide by bin width
    h_mc.Scale(1, "width")
    h_data.Scale(1, "width")
    # compute mean and std
    mean_mc = h_mc.GetMean()
    mean_data = h_data.GetMean()
    std_mc = h_mc.GetStdDev()
    std_data = h_data.GetStdDev()
    # print mean and std
    print(f"Mean of {name}_mc: {mean_mc}, std: {std_mc}")
    print(f"Mean of {name}_data: {mean_data}, std: {std_data}")
    if(do_breakdown):
        for sample in breakdown_list:
            mean = h_mc_breakdown[breakdown_list.index(sample)].GetMean()
            std = h_mc_breakdown[breakdown_list.index(sample)].GetStdDev()
            entries = h_mc_breakdown[breakdown_list.index(sample)].GetEntries()
            print(f"Mean of {name}_{sample}: {mean}, std: {std}, entries: {entries}")

    # create a canvas
    c = ROOT.TCanvas("c", "Canvas", 800, 600)
    # set the histogram styles
    h_mc.SetLineColor(ROOT.kBlue)
    h_mc.SetLineWidth(1)
    h_mc.SetFillColor(ROOT.kBlue)
    h_mc.SetFillStyle(3002)
    h_mc.SetLineStyle(3)
    if do_breakdown:
        # draw the stack
        h_mc_stack.Draw("hist")
    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetLineWidth(2)
    # draw the histograms
    h_mc.Draw(draw_option + " e2same")
    h_data.Draw(draw_option + " same")
    h_data.SetStats(0)
    h_mc.SetStats(0)
    # adjust y axis
    y_max = max(h_mc.GetMaximum(), h_data.GetMaximum())
    h_mc.SetMaximum(y_max * 1.2)
    h_mc.SetMinimum(0)
    if do_breakdown:
        # set the title
        h_mc_stack.SetTitle(name + " MC Breakdown")
        h_mc_stack.GetXaxis().SetTitle(h_mc.GetXaxis().GetTitle())
        h_mc_stack.GetYaxis().SetTitle("Events / bin")
        h_mc_stack.SetMaximum(y_max * 1.2)
        h_mc_stack.SetMinimum(0)
    if do_breakdown:
        # draw the legend
        legend_stack.AddEntry(h_data, "Data", "l")
        legend_stack.Draw()
    else:
        # legend with just mc/data
        legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
        legend.AddEntry(h_mc, "MC", "l")
        legend.AddEntry(h_data, "Data", "l")
        legend.Draw()

    # set log scale if requested
    if logy:
        c.SetLogy()
    # save the canvas
    c.SaveAs(f"{out_dir}/{name}_mcdata_shape.png")
#################################################################################################################################

def plot_qe_notqe(name, draw_option="", logy=False, spline = False):
    h_qe = input_file.Get(name + "_qe")
    if not h_qe:
        print(f"Histogram {name}_qe not found in the file.")
        return
    h_notqe = input_file.Get(name + "_qenot")
    if not h_notqe:
        print(f"Histogram {name}_qenot not found in the file.")
        return
    # divide by bin width
    h_qe.Scale(1, "width")
    h_notqe.Scale(1, "width")
    # create a canvas
    c = ROOT.TCanvas("c", "Canvas", 800, 600)
    # set the histogram styles
    h_qe.SetLineColor(ROOT.kBlue)
    h_qe.SetLineWidth(1)
    h_notqe.SetLineColor(ROOT.kRed)
    h_notqe.SetLineWidth(1)
    # draw the histograms
    h_qe.Draw(draw_option + " l")
    h_notqe.Draw(draw_option + " lsame")
    h_notqe.SetStats(0)
    # adjust y axis
    y_max = max(h_qe.GetMaximum(), h_notqe.GetMaximum())
    h_qe.SetMaximum(y_max * 1.2)
    h_qe.SetMinimum(0)
    h_qe.SetStats(0)
    h_notqe.SetStats(0)

    
    # legend with just qe/notqe
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    legend.AddEntry(h_qe, "QE", "l")
    legend.AddEntry(h_notqe, "Not QE", "l")
    
    legend.Draw()

    # find mean, std, entries and peak 
    mean_qe = h_qe.GetMean()
    mean_notqe = h_notqe.GetMean()
    std_qe = h_qe.GetStdDev()
    std_notqe = h_notqe.GetStdDev()
    entries_qe = h_qe.GetEntries()
    entries_notqe = h_notqe.GetEntries()
    peak_qe = h_qe.GetBinCenter(h_qe.GetMaximumBin())
    peak_notqe = h_notqe.GetBinCenter(h_notqe.GetMaximumBin())
    print(f"Mean of {name}_qe: {mean_qe}, std: {std_qe}, entries: {entries_qe}, peak: {peak_qe}")
    print(f"Mean of {name}_qenot: {mean_notqe}, std: {std_notqe}, entries: {entries_notqe}, peak: {peak_notqe}")
    
    # set log scale if requested
    if logy:
        c.SetLogy()
    
    if spline:
        h_qe_spline = ROOT.TSpline3(h_qe)
        h_qe_spline.SetLineColor(ROOT.kBlue)
        h_qe_spline.SetLineWidth(1)
        h_qe_spline.Draw(draw_option + " same")
        h_notqe_spline = ROOT.TSpline3(h_notqe)
        h_notqe_spline.SetLineColor(ROOT.kRed)
        h_notqe_spline.SetLineWidth(1)
        h_notqe_spline.Draw(draw_option + " same")
    
    # save the canvas
    c.SaveAs(f"{name}_qe_notqe.png")

#################################################################################################################################

def project_MnvH2D_to_MnvH1D(h2d, axis):
    """
    Projects a MnvH2D histogram to a MnvH1D histogram along the specified axis.
    """
    if axis == 0:
        h1d = h2d.ProjectionX()
        h1d.SetTitle(h2d.GetXaxis().GetTitle())
    elif axis == 1:
        h1d = h2d.ProjectionY()
        h1d.SetTitle(h2d.GetYaxis().GetTitle())
    else:
        raise ValueError("Axis must be 0 (X) or 1 (Y).")
    return h1d
    


################################################################################################################


def disown_mnvh2d(obj):
    ROOT.SetOwnership(obj, False)
    for key in obj.GetVertErrorBandNames():
        band = obj.GetVertErrorBand(key)
        ROOT.SetOwnership(band, False)
        for h in band.GetHists():
            ROOT.SetOwnership(h, False)
    for key in obj.GetLatErrorBandNames():
        band = obj.GetLatErrorBand(key)
        ROOT.SetOwnership(band, False)
        for h in band.GetHists():
            ROOT.SetOwnership(h, False)



def obtain_systematics_2D(h_mc, error_band_names):
    """
    Obtain the systematic error bars for a 2D histogram and for one specific systematic.
    :param h_mc: The histogram to obtain the systematics for.
    :param error_band_name: The name of the "error band", i.e. the systematic to vary.
    :return: Projections of the mean value histogram in x and y directions with systematics error bands for the 
                specified systematic. 
    """

    print("Computing systematics error bands for: ", error_band_names)
    cols = h_mc.GetNbinsY()
    rows = h_mc.GetNbinsX()
    
    CentralValues = [[0 for _ in range(cols)] for _ in range(rows)]
    Errors = [[0 for _ in range(cols)] for _ in range(rows)]

    total_universes = 0
    if isinstance(error_band_names, str):
        if error_band_names == "all":
            # Get all error band names
            lat_error_band_names = h_mc.GetLatErrorBandNames()
            vert_error_band_names = h_mc.GetVertErrorBandNames()
            print(type(lat_error_band_names))
            error_band_names = list(lat_error_band_names) + list(vert_error_band_names)
        elif error_band_names == "vert":
            error_band_names = list(h_mc.GetVertErrorBandNames())
        elif error_band_names == "lat":
            error_band_names = list(h_mc.GetLatErrorBandNames())
        else:
            error_band_names = [error_band_names]
    for error_band_name in error_band_names:
        vertical_bool = False
        lateral_bool = False
        # automatically determine if the error band is vertical or lateral
        error_band = h_mc.GetVertErrorBand(error_band_name) 
        if error_band:
            vertical_bool = True
        else:
            error_band = h_mc.GetLatErrorBand(error_band_name)
            if error_band:
                lateral_bool = True
        if not vertical_bool and not lateral_bool:
            print(f"Error: Error band {error_band_name} not found for this histogram.")
            continue
        # Loop over all universes in the error band
        nUniverses = error_band.GetNHists()
        if vertical_bool:
            print(f"Processing vertical error band: {error_band_name}: {nUniverses} universes")
        if lateral_bool:
            print(f"Processing lateral error band: {error_band_name}: {nUniverses} universes")
        total_universes += nUniverses
        for i in range(nUniverses):
            h_thisUniverse = error_band.GetHist(i)
            if not h_thisUniverse:
                print(f"Error: Universe {i} not found in the error band {error_band_name}.")
                continue
            for iBinx in range(h_thisUniverse.GetNbinsX()):
                for iBiny in range(h_thisUniverse.GetNbinsY()):
                    binContent = h_thisUniverse.GetBinContent(iBinx+1, iBiny+1)
                    # print(f"Bin ({iBinx+1}, {iBiny+1}): {binContent}")
                    CentralValues[iBinx][iBiny] += binContent
                    Errors[iBinx][iBiny] += binContent**2
            # print(f"Universe {i} processed.")

    x_bin_edges = np.array([h_mc.GetXaxis().GetBinLowEdge(i) for i in range(1, h_mc.GetNbinsX() + 2)])
    y_bin_edges = np.array([h_mc.GetYaxis().GetBinLowEdge(i) for i in range(1, h_mc.GetNbinsY() + 2)])
    # Calculate the mean and std
    for iBinx in range(rows):
        for iBiny in range(cols):
            CV_fromCVHisto = h_mc.GetBinContent(iBinx+1, iBiny+1)
            CentralValues[iBinx][iBiny] /= total_universes
            if (abs(CV_fromCVHisto - CentralValues[iBinx][iBiny]) > 5):
                print("error band name: ", error_band_name)
                print(f"Bin ({iBinx}, {iBiny}): Central Value from MnvH2D: {CV_fromCVHisto}, Calculated Central Value: {CentralValues[iBinx][iBiny]}")
            Errors[iBinx][iBiny] = (Errors[iBinx][iBiny] / total_universes - CentralValues[iBinx][iBiny]**2)
            if Errors[iBinx][iBiny] < 0:
                print(f"Negative error for bin ({iBinx}, {iBiny}): {Errors[iBinx][iBiny]} - err band: {error_band_name}")
                Errors[iBinx][iBiny] = 0
            Errors[iBinx][iBiny] = Errors[iBinx][iBiny]**0.5
    # Create a new histogram to store the mean and error
    
    h_mean = ROOT.TH2D("h_mean", "Mean and error", x_bin_edges.__len__()-1, x_bin_edges, y_bin_edges.__len__()-1, y_bin_edges)
    # printout binning of h_mean
    # print("Binning of h_mean:")
    # for iBinx in range(h_mean.GetNbinsX()):
    #     print(f"X: Bin {iBinx+1}: {h_mean.GetXaxis().GetBinLowEdge(iBinx+1)} to {h_mean.GetXaxis().GetBinLowEdge(iBinx+2)}")
    # for iBiny in range(h_mean.GetNbinsY()):
    #     print(f"Y: Bin {iBiny+1}: {h_mean.GetYaxis().GetBinLowEdge(iBiny+1)} to {h_mean.GetYaxis().GetBinLowEdge(iBiny+2)}")
    for iBinx in range(h_mean.GetNbinsX()):
        for iBiny in range(h_mean.GetNbinsY()):
            h_mean.SetBinContent(iBinx+1, iBiny+1, CentralValues[iBinx][iBiny])
            h_mean.SetBinError(iBinx+1, iBiny+1, Errors[iBinx][iBiny])
    h_mean.SetName("h_mean")

    # Projection gives segfault in batch mode. Not sure why.
    # ROOT.gROOT.SetBatch(0)
    projx = h_mean.ProjectionX()
    projy = h_mean.ProjectionY()
    projx.Scale(1,"width")
    projy.Scale(1,"width")

    return projx, projy


################################################################################################################

def print_error_bands(h_mc):
    """
    Print the names of the error bands present in a histogram.
    Here "error band" represents one systematic uncertainty.
    :param h_mc: The histogram to print the error bands for.
    """
    lat_names = h_mc.GetLatErrorBandNames()
    vert_names = h_mc.GetVertErrorBandNames()
    for name in vert_names:
        error_band = h_mc.GetVertErrorBand(name)
        nUniverses = error_band.GetNHists()
        print(f"Vert error band {name} has {nUniverses} universes")
    for name in lat_names:
        error_band = h_mc.GetLatErrorBand(name)
        nUniverses = error_band.GetNHists()
        print(f"Lat error band {name} has {nUniverses} universes")

def get_list_of_lat_error_bands(h_mc):
    """
    Get the list of lateral error bands present in a histogram.
    Here "error band" represents one systematic uncertainty.
    :param h_mc: The histogram to print the error bands for.
    :return: A list of lateral error band names.
    """
    lat_names = h_mc.GetLatErrorBandNames()
    list_of_names = [str(name) for name in lat_names]
    return list_of_names

def get_list_of_vert_error_bands(h_mc):
    """
    Get the list of vertical error bands present in a histogram.
    Here "error band" represents one systematic uncertainty.
    :param h_mc: The histogram to print the error bands for.
    :return: A list of vertical error band names.
    """
    vert_names = h_mc.GetVertErrorBandNames()
    print(type(vert_names))
    list_of_names = [str(name) for name in vert_names]
    return list_of_names

def set_my_style():
    myStyle = ROOT.TStyle("myStyle", "Custom style")

    # Margins
    myStyle.SetPadBottomMargin(0.15)
    myStyle.SetPadLeftMargin(0.2)
    myStyle.SetPadTopMargin(0.15)
    myStyle.SetPadRightMargin(0.2)

    # Axis titles
    myStyle.SetTitleFont(42, "XYZ")
    myStyle.SetTitleSize(0.06, "XYZ")
    myStyle.SetTitleOffset(1.0, "XZ")
    myStyle.SetTitleOffset(1.4, "Y")

    # Palette
    # NRGBs = 3
    # NCont = 255
    # # Define endpoints in hex (teal, black, magenta)
    # colors = ["#008080", "#000000", "#AA00FF"]  

    # # Use matplotlib's perceptually uniform interpolation
    # cmap = mpl.colors.LinearSegmentedColormap.from_list("teal_magenta", colors, NCont)

    # red   = []
    # green = []
    # blue  = []
    # stops = np.linspace(0, 1, len(colors))

    # for i, c in enumerate(colors):
    #     rgb = mpl.colors.to_rgb(c)
    #     red.append(rgb[0])
    #     green.append(rgb[1])
    #     blue.append(rgb[2])
    
    # ROOT.TColor.CreateGradientColorTable(NRGBs,
    #                                      array('d', stops),
    #                                      array('d', red),
    #                                      array('d', green),
    #                                      array('d', blue),
    #                                      NCont)
    # myStyle.SetNumberContours(NCont)
    myStyle.SetPalette(ROOT.kViridis)
    
    # Plot title 
    myStyle.SetTitleFont(42, "")
    myStyle.SetTitleSize(0.06, "")

    ROOT.TGaxis.SetMaxDigits(3)  # force scientific if labels exceed 10^3

    # Axis labels
    myStyle.SetLabelSize(0.05, "XYZ")

    # Apply the style globally
    ROOT.gROOT.SetStyle("myStyle")
    ROOT.gROOT.ForceStyle()


def get_pot_scale(input_file):
    input_file = ROOT.TFile.Open(input_file)
    pot = ROOT.TVector2(input_file.Get("pot"))
    #potscale data/mc
    print(f"POT data: {pot.X()}, POT mc: {pot.Y()}")
    potscale = pot.X()/pot.Y()
    return potscale