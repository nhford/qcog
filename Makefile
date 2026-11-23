NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17
NVCC_HOST_COMPILER ?= g++-11
PYTHON ?= python3
INPUT ?= benchmarks/small/tof_3.qasm

.PHONY: all run draw
.DELETE_ON_ERROR:

all: qcog

qcog: main.cpp circuit.cpp qasm.cpp optimizer.cu circuit.h qasm.h optimizer.h
	$(NVCC) -ccbin $(NVCC_HOST_COMPILER) $(NVCCFLAGS) main.cpp circuit.cpp qasm.cpp optimizer.cu -o $@

run: qcog
	./qcog -f "$(INPUT)" > output.qasm

draw: run
	$(PYTHON) draw_circuit.py output.qasm circuit.png

CXX ?= c++
.PHONY: test test-gpu test-all test-commutation
test: tests/qasm_test
	./tests/qasm_test benchmarks/small/tof_*.qasm benchmarks/small/vqe_4.qasm

tests/qasm_test: tests/qasm_test.cpp circuit.cpp qasm.cpp circuit.h qasm.h
	$(CXX) -std=c++17 -Wall -Wextra -I. circuit.cpp qasm.cpp tests/qasm_test.cpp -o $@

test-gpu: tests/gpu_test tests/commutation_test
	./tests/gpu_test benchmarks/hadamard/*.qasm
	./tests/commutation_test

test-commutation: tests/commutation_test
	./tests/commutation_test

test-all: test test-gpu

tests/gpu_test: tests/gpu_test.cu circuit.cpp qasm.cpp optimizer.cu circuit.h qasm.h optimizer.h
	$(NVCC) -ccbin $(NVCC_HOST_COMPILER) $(NVCCFLAGS) -I. circuit.cpp qasm.cpp optimizer.cu tests/gpu_test.cu -o $@

tests/commutation_test: tests/commutation_test.cu circuit.cpp qasm.cpp optimizer.cu circuit.h qasm.h optimizer.h
	$(NVCC) -ccbin $(NVCC_HOST_COMPILER) $(NVCCFLAGS) -I. circuit.cpp qasm.cpp tests/commutation_test.cu -o $@
