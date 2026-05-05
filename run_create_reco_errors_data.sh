#!/bin/bash

in_dir="/eos/experiment/neutplatform/t2knd280/lgiannes/Minerva_tuples/PsiPrimeAnalysisOutputs_minervame1M"
mig_file="${in_dir}/Migration_light.root" 
ana_file="${in_dir}/Analysis_histograms.root" 
# ana_file="${in_dir}/Selection_Signal_light.root" 
eff_file="${in_dir}/Efficiency_light_noPseudoeff.root"


python3 ${MINERVAPYTHON}/create_reco_errors_data.py \
                              --migration "$mig_file" \
                              --main_file "$ana_file"  \
                              --efficiency "$eff_file" \
                              --lam 1.e-5 \
