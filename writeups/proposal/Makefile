LATEX=pdflatex
BIBTEX=bibtex
TARGET_BASE=proposal
TARGET_PDF=$(TARGET_BASE).pdf
TARGET_TEX=$(TARGET_BASE).tex

all: $(TARGET_PDF)

$(TARGET_PDF): $(TARGET_TEX) references.bib
	$(LATEX) $(TARGET_TEX)
	$(BIBTEX) $(TARGET_BASE)
	$(LATEX) $(TARGET_TEX)
	$(LATEX) $(TARGET_TEX)

clean:
	rm *.brf *.lof *.out *.aux *.bbl *.blg *.log *.toc *.dvi $(TARGET_PDF)

