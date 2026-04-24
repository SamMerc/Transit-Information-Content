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
        chains="src/data/Fig2_Storage/16.0ppm/Seed80/chains.npy",
        chi2="src/data/Fig2_Storage/16.0ppm/Seed80/chi2_chain.npy",
        logprob="src/data/Fig2_Storage/16.0ppm/Seed80/logprob.npy",
    output:
        "src/tex/figures/Fig2.pdf"
    cache: True
    script:
        "src/scripts/Fig2_plot.py"


rule Fig3_run:
    input:
        script="src/scripts/Fig3_run.py",
        chains="src/data/Fig2_Storage/16.0ppm/Seed80/chains.npy",
        chi2="src/data/Fig2_Storage/16.0ppm/Seed80/chi2_chain.npy",
        logprob="src/data/Fig2_Storage/16.0ppm/Seed80/logprob.npy",
    output:
        "src/data/Fig3_Storage/chi2_r_r.npy"
    cache: True
    script:
        "src/scripts/Fig3_run.py"


rule Fig3:
    input:
        script="src/scripts/Fig3_plot.py",
        chi2="src/data/Fig3_Storage/chi2_r_r.npy"
    output:
        "src/tex/figures/Fig3.pdf"
    cache: True
    script:
        "src/scripts/Fig3_plot.py"

rule Fig5:
    input:
        script="src/scripts/Fig5_plot.py",
        results="src/data/Fig5_Storage/mps1/results.npz",
    output:
        "src/tex/figures/Fig5.pdf"
    cache: True
    script:
        "src/scripts/Fig5_plot.py"

rule Appendix2:
    input:
        script="src/scripts/Appendix2_plot.py",
        results="src/data/Fig5_Storage/mps1/results.npz",
    output:
        "src/tex/figures/Appendix2.pdf"
    cache: True
    script:
        "src/scripts/Appendix2_plot.py"

rule Appendix1:
    input:
        script="src/scripts/Appendix1_plot.py",
        chi2="src/data/Fig3_Storage/chi2_r_r.npy"
    output:
        "src/tex/figures/Appendix1.pdf"
    cache: True
    script:
        "src/scripts/Appendix1_plot.py"
