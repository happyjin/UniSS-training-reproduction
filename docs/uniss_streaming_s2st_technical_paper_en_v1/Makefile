TECTONIC ?= /opt/dlami/nvme/jasonleeeli/tools/tectonic-0.15.0/tectonic

.PHONY: all clean

all:
	mkdir -p build
	$(TECTONIC) --keep-logs --keep-intermediates --outdir build main.tex

clean:
	rm -f build/main.aux build/main.bbl build/main.blg build/main.log build/main.out build/main.pdf
