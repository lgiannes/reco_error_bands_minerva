#!/bin/bash


# in_dir="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/PsiPrimeAnalysisOutputs_minervame1B"
# mig_file="${in_dir}/Migration_light.root" 
# sel_file="${in_dir}/Selection_Signal_light.root" 
# eff_file="${in_dir}/Efficiency_light_noPseudoeff.root"

in_dir="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/MergedPlaylists/CleanAllNewCalib"
mig_file="${in_dir}/Migration_lightABCDEFGLMbisNOP.root" 
sel_file="${in_dir}/Selection_Signal_lightABCDEFGLMbisNOP.root" 
eff_file="${in_dir}/Efficiency_lightABCDEFGLMbisNOP.root"

# mig_file="${in_dir}/Migration_light_bdtRew.root" 
# sel_file="${in_dir}/Selection_Signal_light_bdtRew.root" 
# eff_file="${in_dir}/Efficiency_light_bdtRew_noPseudoeff.root"


out_dir="${in_dir}/Rdagger_from_refolded_TEST/"
mkdir -p "${out_dir}"

python3 ${MINERVAPYTHON}/create_reco_errors.py \
                              --migration "$mig_file" \
                              --selection "$sel_file"  \
                              --efficiency "$eff_file" \
                              --outdir "$out_dir" \
                              --rcond 1.e-3 \
                              --lam 1.e-6 \
                              --use_truth
