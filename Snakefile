rule Fig1:
    input:
        script="src/scripts/Fig1_plot.py",
        data="src/data/Fig1_Storage/processed_data_cache.pkl"
    output:
        "tex/figures/Fig1.pdf"
    script:
        "src/scripts/Fig1_plot.py"


rule Fig2:
    input:
        script="src/scripts/Fig2_plot.py",
        data="src/data/Fig2_Storage"
    output:
        "tex/figures/Fig2.pdf"
    script:
        "src/scripts/Fig2_plot.py"


rule Fig3_run:
    input:
        script="src/scripts/Fig3_run.py",
        data="src/data/Fig2_Storage"
    output:
        "src/data/Fig3_Storage/chi2_r_r.npy"
    script:
        "src/scripts/Fig3_run.py"


rule Fig3:
    input:
        script="src/scripts/Fig3_plot.py",
        data="src/data/Fig3_Storage",
        chi2="src/data/Fig3_Storage/chi2_r_r.npy"
    output:
        "tex/figures/Fig3.pdf",
        "tex/figures/Fig5.pdf"
    script:
        "src/scripts/Fig3_plot.py"

rule Appendix1:
    input:
        script="src/scripts/Appendix1_plot.py",
        data="src/data/Appendix1_Storage"
    output:
        "tex/figures/Appendix1.pdf",
    script:
        "src/scripts/Appendix1_plot.py"