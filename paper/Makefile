# Build the paper with tectonic (preferred) or latexmk if tectonic is absent.
TECTONIC ?= $(shell command -v tectonic 2>/dev/null || echo $(HOME)/.local/bin/tectonic)
MAIN     := main
BUILD    := build
SRC      := $(MAIN).tex refs.bib $(wildcard sections/*.tex) $(wildcard figures/*)

.PHONY: all clean todo watch

all: $(BUILD)/$(MAIN).pdf

$(BUILD)/$(MAIN).pdf: $(SRC)
	@mkdir -p $(BUILD)
	@if [ -x "$(TECTONIC)" ]; then \
		"$(TECTONIC)" -X compile --keep-logs --outdir $(BUILD) $(MAIN).tex; \
	elif command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -outdir=$(BUILD) $(MAIN).tex; \
	else \
		echo "need tectonic or latexmk"; exit 1; \
	fi

# list every placeholder still in the draft
todo:
	@grep -n -E '\\(todo|pending)\{' $(MAIN).tex sections/*.tex || echo "no placeholders"

clean:
	rm -rf $(BUILD)
