ANDi Research Experiments — LaTeX source

FILES
- ANDi_Research_Experiments.tex : complete source for the 13-page updated PDF
- noise_schedule.png           : chart included from the LaTeX file

BUILD
Compile using pdfLaTeX (run twice to update cross-references and page counts):

    pdflatex -interaction=nonstopmode -halt-on-error ANDi_Research_Experiments.tex
    pdflatex -interaction=nonstopmode -halt-on-error ANDi_Research_Experiments.tex

The source uses standard LaTeX packages including TikZ, tcolorbox, amsmath,
graphicx, listings, hyperref, and others listed in the preamble.
The chart image must remain in the same directory as the .tex file.
