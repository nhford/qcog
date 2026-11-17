#pragma once
#include "circuit.h"
#include <iosfwd>

// Host-only OpenQASM 2 loader/writer; angles are stored in units of pi.
Circuit read_circuit(std::istream &input);
void write_circuit(std::ostream &output, const Circuit &circuit);
