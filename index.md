---
layout: default
title: "QCOG: Milestone Report"
---

# QCOG: Quantum Circuit Optimization on GPUs

Group Members: Noah Ford, Changxiang Gao

https://nhford.github.io/qcog/

15-618 Final Project Milestone Report

## Project Summary

We implement the rule-based quantum optimizer from Nam et al. as sweeps of data-parallel local optimizations. The research question is to adapt the sequential algorithm to a data parallel model, and to compare efficiency as well as optimization performance against a sequential version and other optimizers.

## Updated Project Schedule

<!-- TODO: Replace this with an up-to-date schedule in half-week increments. Include completed work, tasks for the remaining weeks, and an owner for every task. -->

| Date       | Task                                                            | Owner            | Status      |
| ---------- | --------------------------------------------------------------- | ---------------- | ----------- |
| Fri, 11/6  | Project Idea proposal with Dmitros                              | Changxiang       | Done        |
| Mon, 11/9  | Project Proposal Due                                            | Noah, Changxiang | Done        |
| Wed, 11/11 | Read all of relevant background papers: Nam et al., POPQC, VOQC | Both             | Done        |
| Mon, 11/16 | Build representation of circuits for CUDA kernels to pass       | Both             | Done        |
| Wed, 11/18 | Hadamard Reduction local rewrite kernel                         | Noah             | Done        |
| Fri, 11/20 | Single-qubit transformation rewrites: Match                     | Changxiang       | Done        |
| Fri, 11/20 | Single-qubit transformation rewrites: Apply                     | Noah             | Done        |
| Mon, 11/23 | Midterm Project Proposal Due                                    | Both             | Done        |
| Fri, 11/27 | NOT Propagation                                                 | Noah             | In Progress |
| Mon, 11/30 | Phase Polynomial Reductions                                     | Changxiang       | TODO        |
| Wed, 12/2  | Finalize responsibilities of host vs kernels                    | Both             | TODO        |
| Fri, 12/4  | Finish profiling our implementation vs POPQC                    | Noah             | TODO        |
| Fri, 12/4  | Finish profiling our implementation vs VOQC                     | Changxiang       | TODO        |
| Mon, 12/7  | Buffer space for unfinished tasks / Stretch: Heavy              | TODO             | TODO        |
| Fri, 12/11 | Project Presentation Day                                        | TODO             | TODO        |

## Work Completed So Far

<!-- TODO: Summarize completed work in one or two paragraphs. Include what is implemented, tested, and learned since the initial proposal. -->

First we were able to scaffold the pipeline to parse the .qasm file circuits and then represent the circuit as a struct of arrays. This was adapted from the POPQC repository which is written in Rust. The majority of the work in this step was in our conscience design of the fields necessary in the circuit (gate type, qubit indices, neighbor pointers) but additionally we added the representation of gate angles as components of pi and a real number. The POPQC implementation only just parses "pi/2" as a real number, which makes it more challenging to convert back exactly to a fraction of pi on output.

Additionally, we were able to implement the first three of four optimizations outlined by the Nam algorithm: Hadamard reduction patterns, single qubit local rewrites, and two qubit local rewrites. This constituted the majority of our development time. The Hadamard reductions required one thread per gate to check whether it was a Hadamard that started one of three given optimizable patterns. The single qubit local rewrites looks at if a gate is cancellable with its neighbor (so inverses) or after swapping with commutating neighbors. This rewrite is also a CUDA Kernel. The two qubit rewrite has proved to be the most challenging because the CNOT gate is harder to check neighbors, as their are two wires it lives on, but we have finished it as well.

## Progress Toward Goals and Deliverables

### Original Goals

Complete a parallelized version of the Nam et al. rule set of optimizations such that it can optimize any given circuit written in the Nam gate set. We believe this is achievable given there is already a framework for a multicore and sequential implementation of this rule set to draw from.

We would like to then demonstrate the timing performance and optimizer effectiveness of our implementation against that of VOQC and other optimizers. This would end up being the final analysis of the deliverable – likely a graph demonstrating the difference between our and others’ implementation.

The quantum group’s POPQC paper will be our target for speed-up times. Meaning, the multi-core speed-up times for each input circuit will be lower bound we shoot for with our GPU version. The POPQC paper also maintained near perfect optimization quality on the entire suite of circuits it optimized, so we would hope to remain within a margin of error for our replication.

The mode of the Nam et al. round of sequences we plan to build is its “Light” mode which is a computationally simpler succession of rules, but a stretch would be to add support for its “Heavy” mode which involves rule #5 mentioned above.

### Updated Goals for the Poster Session

<!-- TODO: State which original deliverables remain achievable, including the stretch goal, explain any changes, and list the revised goals for the poster session. -->

Our deliverable of the "Light Sequence" implementation of the Nam paper remains achievable. After we finish the fourth rewrite rule (phase polynomial cancellation), we should be able to benchmark our solution against that of POPQC and others.

The stretch goal of the "Heavy" sequence is also very much in reach as it would require just one additional rewrite, or kernel. However, for now, that remains a stretch.

## Preliminary Results

<!-- TODO: Add preliminary results, such as correctness checks, gate-count/depth changes, performance measurements, and comparisons with reference optimizers. Include figures or tables when available. -->

Our first three rewrite rules have passed small test circuits. But all of the input circuits we plan to optimize require the fourth rewrite to start to unlock actual reductions in gate counts. So we have yet to observe a metric in our final deliverable.

Meaning, this benchmarking has not started yet against the other optimizers, but we have added the set of circuits we plan to consider to our repository.

## Poster Session Plan

<!-- TODO: Say what will be shown at the poster session: a demo, graph, or another presentation format. -->

We plan to have graphs of our optimizer speed-up and optimizer quality against existing models.

We also may consider a visual of how the threads explore the circuit, but that remains a stretch goal.

## Issues and Remaining Unknowns

<!-- TODO: List the main concerns, unresolved technical questions, and any resource or access issues. -->

The main concern is just iterating on the circuit struct format which is currently a struct of arrays which represents the entire series of gates as one flat list. The reference POPQC paper maintains this format but also a DAG, and also a layered representation where each layer is a vertical cut of the circuit.

Once we are able to benchmark against other circuits, we'll be better positioned to iterate on a design that minimizes load latency.

## Background and Method

The bottleneck of quantum computation is largely agreed upon to be error correction, which remains a challenging software and hardware problem. The primary lever to make quantum computation and simulation more achievable thus far has been reducing the circuit size and depth such that the likelihood of error is reduced, or in other words – optimizing the circuit’s number of gates, depth of gates, etc.

<img width="417" height="291" alt="image" src="https://github.com/user-attachments/assets/6e69f029-0104-4190-9cc7-5ebab25a945b" />

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

## Implementation Challenges

The underlying problem of local quantum circuit optimization is known to be "embarrassingly parallel”. However, we have to actually take advantage of this embarrassment. Meaning, we need to first represent and organize the circuit in a structure easily translatable to GPUs kernels. We need to translate quantum circuit logic into CUDA operations on datatypes we build.

We need to draft kernels which perform each of the optimization steps above, which require complexity in representing and navigating the rule-based search space.

We then need to actually schedule the sequence of passes with iterations of kernel calls.

To my understanding, the workload would have minimal dependencies within a single rule of optimization. But between optimization rules, there needs to be complete synchronization. The memory access characteristics are highly localized for the first three rules but need to stretch more far-ranging for the latter two. Divergent execution would occur all over the place based on whether at a given gate or not, the process of an optimization can begin.

Constraints from the system’s properties which make the mapping the workload challenging come from the fact that the optimizer needs work on arbitrary input circuits, which may have vastly differing distributions of gates versus number of qubits. Additionally, some optimization rules may have greater effect than others depending on the input circuit construction.

<!-- TODO: Describe how implementation has addressed these challenges so far and what remains. -->

## Resources and Platform

Nam et al. provides the optimization algorithm for their set of fixed rules. VOQC (Verified Optimizer for Quantum Circuits) is an open source sequential optimizer which implements this set of rules. Additionally, IBM’s Quantum research group has a multicore implementation of VOQC, which is written in Rust.

We would use each of these three sources as our primary reference for implementing our CUDA-optimizer.

Machine access: we would benefit from the usage of a GPU. Because we’re

The language of development would be CUDA given this is built for GPUs.

The specific GPU we run on can be flexible, but for now, we expect an A100.

<!-- TODO: Update the actual hardware and software resources used, if different from the original plan. -->

## Original Proposal

The [initial project proposal]({{ '/initial-proposal.html' | relative_url }}) contains the original site content as submitted.
