OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
rz(-pi/2) q[0];
h q[0];
