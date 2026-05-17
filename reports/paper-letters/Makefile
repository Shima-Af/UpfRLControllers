# Build the IEEE Networking Letters PDF.
#
# Usage:
#   make                # build paper.pdf
#   make clean          # remove build artefacts (keeps paper.pdf, sources, figure)
#   make distclean      # also remove paper.pdf
#
# Requirements:
#   pdflatex (TeX Live) — IEEEtran.cls is bundled in this directory so no
#                          texlive-publishers package is required.

LATEX  := TEXINPUTS=.: pdflatex -interaction=nonstopmode -halt-on-error
SRC    := paper.tex
PDF    := paper.pdf

.PHONY: all clean distclean

all: $(PDF)

$(PDF): $(SRC) figure1.pdf IEEEtran.cls
	$(LATEX) $(SRC) >/dev/null
	$(LATEX) $(SRC) >/dev/null
	@grep -E "Overfull|Undefined" paper.log || true
	@echo "Built $(PDF) ($$(pdfinfo $(PDF) | awk '/^Pages:/ {print $$2}') pages)"

clean:
	rm -f paper.aux paper.log paper.out paper.toc paper.bbl paper.blg

distclean: clean
	rm -f $(PDF)
