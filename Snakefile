rule Fig1:
    input:
        script="src/scripts/Fig1_plot.py",
    output:
        "src/tex/figures/Fig1.pdf"
    cache: True
    script:
        "src/scripts/Fig1_plot.py"


rule Fig2:
    input:
        script="src/scripts/Fig2_plot.py",
    output:
        "src/tex/figures/Fig2.pdf"
    cache: True
    script:
        "src/scripts/Fig2_plot.py"


rule Fig3_run:
    input:
        script="src/scripts/Fig3_run.py",
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


rule Appendix1:
    input:
        script="src/scripts/Appendix1_plot.py",
        chi2="src/data/Fig3_Storage/chi2_r_r.npy"
    output:
        "src/tex/figures/Appendix1.pdf"
    cache: True
    script:
        "src/scripts/Appendix1_plot.py"
