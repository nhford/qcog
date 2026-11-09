---
layout: default
title: "QCOG: Quantum Circuit Optimization on GPUs"
---

# QCOG: Quantum Circuit Optimization on GPUs

15-618 Final Project Proposal

## URL

todo

## Summary

We implement the rule-based quantum optimizer from Nam et al. as sweeps of data-parallel local optimizations. The research question is to adapt the sequential algorithm to a data parallel model, and to compare efficiency as well as optimization performance against a sequential version and other optimizers.

## Background

The bottleneck of quantum computation is largely agreed upon to be error correction, which remains a challenging software and hardware problem. The primary lever to make quantum computation and simulation more achievable thus far has been reducing the circuit size and depth such that the likelihood of error is reduced, or in other words – optimizing the circuit’s number of gates, depth of gates, etc.

TODO: include image of circuit

Quantum circuits are represented by qubits which are the input and output datatype, and gates which are unitary operations which act upon one or more qubits. The width of a circuit is the number of qubits used, and the depth is the longest number of gates that need to be executed in sequence.

The Nam algorithm attempts to reduce the depth of the circuit and the total gate count by performing cancelling operations, commutative operations, and merging operations.

These full rules are described in Nam et al., but summarized here:

| #   | Optimization                   | Runtime (gates) | Parallelization                                                                                                   |
| --- | ------------------------------ | --------------- | ----------------------------------------------------------------------------------------------------------------- |
| 1   | Hadamard Reduction             | `O(g)`          | Each hadamard operation can verify whether it can be reduced independently                                        |
| 2   | Single-qubit gate cancellation | `O(g^2)`        | Each qubit checks whether each of its gates can be commuted such that they can be cancelled with inverse          |
| 3   | Two-qubit gate cancellation    | `O(g^2)`        | Two qubit-pairs can search each of their CNOT gates (only 2-qubit gate in gate set) in search of cancels          |
| 4   | Rotation merging               | `O(g log g^2)`  | For each rotation gate, check to see if it can be combined with adjacent rotation gate                            |
| 5   | Floating R_z gates             | `O(g^3)`        | This has a very large search space for moving Rotation gates between other sub-circuits within the entire circuit |

The optimizer then runs in rounds of these rule applications above in a sequence. The original paper cites two such sequences, the first characterized as the “Light” version: 1, 3, 2, 3, 1, 2, 4, 3, 2.

## The Challenge

The underlying problem of local quantum circuit optimization is known to be "embarrassingly parallel”. However, we have to actually take advantage of this embarrassment. Meaning, we need to first represent and organize the circuit in a structure easily translatable to GPUs kernels. We need to translate quantum circuit logic into CUDA operations on datatypes we build.

We need to draft kernels which perform each of the optimization steps above, which require complexity in representing and navigating the rule-based search space.

We then need to actually schedule the sequence of passes with iterations of kernel calls.

To my understanding, the workload would have minimal dependencies within a single rule of optimization. But between optimization rules, there needs to be complete synchronization. The memory access characteristics are highly localized for the first three rules but need to stretch more far-ranging for the latter two. Divergent execution would occur all over the place based on whether at a given gate or not, the process of an optimization can begin.

Constraints from the system’s properties which make the mapping the workload challenging come from the fact that the optimizer needs work on arbitrary input circuits, which may have vastly differing distributions of gates versus number of qubits. Additionally, some optimization rules may have greater effect than others depending on the input circuit construction.

## Resources

Nam et al. provides the optimization algorithm for their set of fixed rules. VOQC (Verified Optimizer for Quantum Circuits) is an open source sequential optimizer which implements this set of rules. Additionally, IBM’s Quantum research group has a multicore implementation of VOQC, which is written in Rust.

We would use each of these three sources as our primary reference for implementing our CUDA-optimizer.

Machine access: we would benefit from the usage of a GPU. Because we’re

## Goals and Deliverables

### Plan to Achieve

Complete a parallelized version of the Nam et al. rule set of optimizations such that it can optimize any given circuit written in the Nam gate set. We believe this is achievable given there is already a framework for a multicore and sequential implementation of this rule set to draw from.

We would like to then demonstrate the timing performance and optimizer effectiveness of our implementation against that of VOQC and other optimizers. This would end up being the final analysis of the deliverable – likely a graph demonstrating the difference between our and others’ implementation.

The quantum group’s POPQC paper will be our target for speed-up times. Meaning, the multi-core speed-up times for each input circuit will be lower bound we shoot for with our GPU version. The POPQC paper also maintained near perfect optimization quality on the entire suite of circuits it optimized, so we would hope to remain within a margin of error for our replication.

### Hope to Achieve

The mode of the Nam et al. round of sequences we plan to build is its “Light” mode which is a computationally simpler succession of rules, but a stretch would be to add support for its “Heavy” mode which involves rule #5 mentioned above.

## Platform Choice

The language of development would be CUDA given this is built for GPUs.

The specific GPU we run on remains undecided (TODO).

## Schedule

| Date       | Action Item Due                                                                 |
| ---------- | ------------------------------------------------------------------------------- |
| Fri, 11/6  | Project Idea proposal with Dmitros                                              |
| Mon, 11/9  | Project Proposal Due                                                            |
| Wed, 11/11 | Read all of relevant background papers: Nam et al., POPQC, VOQC                 |
| Fri, 11/13 | Finalize responsibilities of host vs kernels                                    |
| Mon, 11/16 | Build representation of circuits for CUDA kernels to pass                       |
| Wed, 11/18 | Hadamard Reduction local rewrite kernel                                         |
| Fri, 11/20 | Outline single-qubit transformation rewrites                                    |
| Mon, 11/23 | Midterm Project Proposal Due,                                                   |
| Fri, 11/27 | Implement single-qubit transformation rewrites via propagation of altered gates |
| Mon, 11/30 | Outline kernel/host plan for Rotation merging                                   |
| Wed, 12/2  | Implement rotation merging                                                      |
| Fri, 12/4  | Finish profiling our implementation vs VOQC vs POPQC                            |
| Mon, 12/7  | Buffer space for unfinished tasks                                               |
| Fri, 12/11 | Project Presentation Day                                                        |

## 15-618 Professor Meeting

Quantum Circuit Optimization on GPUs: How much of a rule-based optimizer can be made parallel?

We share an interest in doing a quantum-related project. This semester, my professor, Umut Acar, for 15-898 had explained that circuit optimization is an embarrassingly parallel problem.

In particular he and TA Pengyu Liu suggested this potential project idea: adapting the basic rule-based optimizer of Nam to GPUs. Their lab released a paper which adapted a SOTA sequential quantum circuit optimizer, VOQC, to a multi-core CPU solution. This research question would include comparing the performance/speed of our rule-based optimizer to the base VOQC and their lab's multi-core POPQC implementation.

The project would require the following steps:

1. representing the circuit as a data structure to parallelize over
2. writing/organizing the functions for local rewrite optimizations
3. writing/organizing larger propagating rules
4. scheduling the sequence of passes
5. evaluating the correctness/performance on existing benchmark circuits versus existing optimizers

Starting points left pointed out to us were the Nam optimizer rules which has an associated paper, Professor Umut Acar's group's paper on POPQC, VOQC itself.
