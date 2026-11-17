NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17
PYTHON ?= python3
INPUT ?= benchmarks/small/tof_3.qasm

.PHONY: all run draw
.DELETE_ON_ERROR:

all: qcog

qcog: main.cpp circuit.cpp qasm.cpp optimizer.cu circuit.h qasm.h optimizer.h
	$(NVCC) $(NVCCFLAGS) main.cpp circuit.cpp qasm.cpp optimizer.cu -o $@

run: qcog
	./qcog -f "$(INPUT)" > output.qasm

draw: run
	$(PYTHON) draw_circuit.py output.qasm circuit.png

CXX ?= c++
.PHONY: test
test: tests/qasm_test
	./tests/qasm_test benchmarks/small/tof_*.qasm benchmarks/small/vqe_4.qasm

tests/qasm_test: tests/qasm_test.cpp circuit.cpp qasm.cpp circuit.h qasm.h
	$(CXX) -std=c++17 -Wall -Wextra -I. circuit.cpp qasm.cpp tests/qasm_test.cpp -o $@
