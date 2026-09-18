rule Fig1:
    input:
        script="src/scripts/Fig1_plot.py",
        chains="src/data/Fig1_Storage/PLD_3/359.000ppm/Seed100/chains.npy",
        chi2="src/data/Fig1_Storage/PLD_3/359.000ppm/Seed100/chi2_chain.npy",
        logprob="src/data/Fig1_Storage/PLD_3/359.000ppm/Seed100/logprob.npy",
        cache="src/data/Fig1_Storage/processed_data_cache.pkl",
    output:
        "src/tex/figures/Fig1.pdf"
    cache: True
    script:
        "src/scripts/Fig1_plot.py"


rule Fig2:
    input:
        script="src/scripts/Fig2_plot.py",
        fig2_cache="src/data/Fig2_Storage/Fig2_base_processed_cache.pkl",
    output:
        "src/tex/figures/Fig2.pdf"
    cache: True
    script:
        "src/scripts/Fig2_plot.py"


rule Fig4_run:
    input:
        script="src/scripts/Fig4_run.py",
        fig2_cache="src/data/Fig2_Storage/Fig2_base_processed_cache.pkl",
    output:
        "src/data/Fig4_Storage/Seed70/chi2_r_r.npy"
    cache: True
    script:
        "src/scripts/Fig4_run.py"


rule Fig4:
    input:
        script="src/scripts/Fig4_plot.py",
        chi2="src/data/Fig4_Storage/Seed70/chi2_r_r.npy",
        fig2_cache="src/data/Fig2_Storage/Fig2_base_processed_cache.pkl",
        fig4_cache="src/data/Fig4_prerun_Storage/processed_data_cache.pkl",
    output:
        "src/tex/figures/Fig4.pdf"
    cache: True
    script:
        "src/scripts/Fig4_plot.py"

rule Fig3:
    input:
        script="src/scripts/Fig3_plot.py",
        results="src/data/Fig3_Storage/mps1/results.npz",
    output:
        "src/tex/figures/Fig3.pdf"
    cache: True
    script:
        "src/scripts/Fig3_plot.py"

# Note: Fig6_prerun.py builds the per-channel truth + noisy chromatic dataset and fits a
# white light curve (WLC) per star to constrain the orbital parameters, for all five stars
# in one invocation. Fig6_run.py then runs one small MCMC (jaxoplanet + emcee_jax; depth +
# limb-darkening coefficients only, orbital parameters fixed at the WLC best fit) per
# (star, wavelength channel, limb-darkening law) combination -- 5 stars x ~219 channels x
# 3 laws in total -- so it is dispatched externally (e.g. an HPC job array) rather than by
# this Snakefile, exactly like Fig3/Fig4/Fig5_run.py. Fig6_plot.py assembles whichever
# channels have already been run from each star's wav_grid.npz + per-channel summary.npz
# files. Once the full grid has been run and uploaded to Zenodo, this rule's inputs should
# be expanded to list the complete set of per-channel outputs (as Fig3/Fig5's rules do for
# their own Zenodo data).
rule Fig6:
    input:
        script="src/scripts/Fig6_plot.py",
        c5="src/data/Fig6_Storage/C5/prerun.npz",
        c1="src/data/Fig6_Storage/C1/prerun.npz",
        c2="src/data/Fig6_Storage/C2/prerun.npz",
        c7="src/data/Fig6_Storage/C7/prerun.npz",
        c6="src/data/Fig6_Storage/C6/prerun.npz",
    output:
        "src/tex/figures/Fig6.pdf"
    cache: True
    script:
        "src/scripts/Fig6_plot.py"

rule Appendix4:
    input:
        script="src/scripts/Appendix4_plot.py",
        chi2="src/data/Fig4_Storage/Seed70/chi2_r_r.npy",
        fig2_cache="src/data/Fig2_Storage/Fig2_base_processed_cache.pkl",
    output:
        "src/tex/figures/Appendix4.pdf"
    cache: True
    script:
        "src/scripts/Appendix4_plot.py"
        
rule Appendix1:
    input:
        script="src/scripts/Appendix1_plot.py",
        results="src/data/Fig3_Storage/mps1/results.npz",
    output:
        "src/tex/figures/Appendix1.pdf"
    cache: True
    script:
        "src/scripts/Appendix1_plot.py"

rule Appendix3:
    input:
        script="src/scripts/Appendix3_plot.py",
        results="src/data/Fig3_Storage/mps1/results.npz",
    output:
        "src/tex/figures/Appendix3.pdf"
    cache: True
    script:
        "src/scripts/Appendix3_plot.py"

rule Appendix2:
    input:
        script="src/scripts/Appendix2_plot.py",
        results="src/data/Fig3_Storage/mps1/results.npz",
    output:
        "src/tex/figures/Appendix2.pdf"
    cache: True
    script:
        "src/scripts/Appendix2_plot.py"

rule Appendix5:
    input:
        script="src/scripts/Appendix5_plot.py",
        mps1="src/data/Appendix5_Storage/mps1/results.npz",
        mps2="src/data/Appendix5_Storage/mps2/results.npz",
    output:
        "src/tex/figures/Appendix5.pdf"
    cache: True
    script:
        "src/scripts/Appendix5_plot.py"

rule Fig5:
    input:
        script="src/scripts/Fig5_plot.py",
        c0_cache="src/data/Fig5_Storage/C0/processed_data_cache.pkl",
        c2_cache="src/data/Fig5_Storage/C2/processed_data_cache.pkl",
        c3_cache="src/data/Fig5_Storage/C3/processed_data_cache.pkl",
    output:
        "src/tex/figures/Fig5.pdf"
    cache: True
    script:
        "src/scripts/Fig5_plot.py"

rule Appendix6:
    input:
        script="src/scripts/Appendix6_plot.py",
        c4_cache="src/data/Fig5_Storage/C4/processed_data_cache.pkl",
        c7_cache="src/data/Fig5_Storage/C7/processed_data_cache.pkl",
    output:
        "src/tex/figures/Appendix6.pdf"
    cache: True
    script:
        "src/scripts/Appendix6_plot.py"
