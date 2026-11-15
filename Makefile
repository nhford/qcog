NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17
PYTHON ?= python3
INPUT ?= benchmarks/small/tof_3.qasm

.PHONY: all run draw
.DELETE_ON_ERROR:

all: qcog

qcog: main.cpp optimizer.cu circuit.h
	$(NVCC) $(NVCCFLAGS) main.cpp optimizer.cu -o $@

run: qcog
	./qcog -f "$(INPUT)" > output.qasm

draw: run
	$(PYTHON) draw_circuit.py output.qasm circuit.png
