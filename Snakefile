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

# Note: Fig6_run.py runs one joint MCMC (shared orbital parameters + per-wavelength-channel
# depth/limb-darkening coefficients, jaxoplanet + emcee_jax) per (star, limb-darkening law)
# combination -- 3 stars x 3 laws = 9 combinations in total -- so it is dispatched externally
# rather than by this Snakefile, exactly like Fig3/Fig4/Fig5_run.py. Fig6_plot.py assembles
# whichever combinations have already been run from each one's summary.npz. Once the full
# 3x3 grid has been run and uploaded to Zenodo, this rule's inputs should be expanded to list
# the complete set of summary.npz outputs (as Fig3/Fig5's rules do for their own Zenodo data).
rule Fig6:
    input:
        script="src/scripts/Fig6_plot.py",
        m_dwarf_quad="src/data/Fig6_Storage/M_dwarf/PLD_2/summary.npz",
        m_dwarf_cube="src/data/Fig6_Storage/M_dwarf/PLD_3/summary.npz",
        m_dwarf_nlld="src/data/Fig6_Storage/M_dwarf/4NLLD/summary.npz",
        g_dwarf_quad="src/data/Fig6_Storage/G_dwarf/PLD_2/summary.npz",
        g_dwarf_cube="src/data/Fig6_Storage/G_dwarf/PLD_3/summary.npz",
        g_dwarf_nlld="src/data/Fig6_Storage/G_dwarf/4NLLD/summary.npz",
        f_dwarf_quad="src/data/Fig6_Storage/F_dwarf/PLD_2/summary.npz",
        f_dwarf_cube="src/data/Fig6_Storage/F_dwarf/PLD_3/summary.npz",
        f_dwarf_nlld="src/data/Fig6_Storage/F_dwarf/4NLLD/summary.npz",
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
