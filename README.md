# Measurement Based Variational Quantum Circuits

**mVQE** is a Julia library designed to facilitate the implementation of the algorithms introduced in [*Learning Feedback Mechanisms for Measurement-Based Variational Quantum State Preparation*](https://arxiv.org/abs/2411.19914). It leverages the [ITensors.jl](https://github.com/ITensor/ITensors.jl) framework for tensor network computations and integrates seamlessly with [Flux.jl](https://fluxml.ai/) for machine learning components.

## Table of Contents

1. [Features](#features)  
2. [API Breakdown](#api-breakdown)  
   - [ITensorsExtension](#itensorsextension)  
   - [StateFactory](#statefactory)  
   - [Gates](#gates)  
   - [Layers](#layers)  
   - [GirvinProtocol](#girvinprotocol)  
   - [FluxExtension](#fluxextension)  
   - [Circuits](#circuits)  
   - [Optimization](#optimization)  
   - [ITensorsMeasurement](#itensorsmeasurement)  

---

## Features

- **Efficient one-qubit gate application**: A custom `runcircuit` function enables fast forward and backward passes when applying one-qubit gates to a state (MPS).
- **Entropy & mutual information**: Functions for computing various measures of entanglement (von Neumann entropy, mutual information, etc.) on both pure and mixed states.
- **MPS construction**: A suite of functions to initialize commonly used matrix product states.
- **Circuit construction**: Variational circuit components and layers, both standard and measurement-based (with mid-circuit measurements).
- **Hamiltonian expectation**: An efficient `expect` function for computing expectation values w.r.t. a Hamiltonian, optimized for backpropagation.
- **Neural network feedback**: An optional Flux-based approach for feedback and adaptive parameter updates in the mVQE algorithm.
- **Girvin protocol**: An implementation of the measurement-based protocol from [Smith *et al.*, PRX Quantum 4, 020315](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.020315).


## API Breakdown

### ITensorsExtension

- **File**: `src/ITensorsExtension/apply.jl`
- **Key Function**: `runcircuit(psi, circuit; onequbit_gates=true)`
  - Efficiently applies one-qubit unitary gates to an MPS state.
  - Includes a faster backpropagation pathway than the default ITensors methods.
- **Additional Utilities**:
  - Entropy and mutual information calculations for both pure and mixed states.

### StateFactory

- **File**: `src/StateFactory.jl`
- Contains helper functions to create or initialize various matrix product states (e.g., product states, random states, GHZ states, etc.).

### Gates

- **File**: `src/Gates.jl`
- Provides definitions of different quantum gates.
- These gate definitions can be used to build custom circuits with measurement-based or standard gate-based approaches.

### Layers

- **File**: `src/Layers.jl`
- Includes definitions of “layers” for building deeper variational circuits easily.

### GirvinProtocol

- **File**: `src/GirvinProtocol.jl`
- Implements the measurement-based protocol described by [Smith *et al.* (PRX Quantum 4, 020315)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.020315).

### FluxExtension

- **Folder**: `src/FluxExtension/`
- Implements Flux-based recurrent neural network that can be used in the feedback steps of the mVQE algorithm.
- Directly integrates with `Circuits` to enable gradient-based updates and advanced optimization routines.

### Circuits

- **Folder**: `src/Circuits/`
- Houses the definitions of variational circuits used by the mVQE algorithm.
- *MeasurementCircuits* sub-module: Provides circuits with mid-circuit measurement and classical feedback, allowing measurement-based VQE protocols.

### Optimization

- **File**: `src/Optimization.jl`
- Contains the `expect` function for Hamiltonian expectation value computations.
- Optimized to replace ITensors’ standard `inner(psi', H, psi)` call, offering better performance when backpropagating fot the calulcations of gradients.

### ITensorsMeasurement

- **Folder**: `src/ITensorsMeasurement/`
- Implements projective measurements on both pure and mixed states.
- Handles backpropagation through measurement steps, keeping the workflow end-to-end differentiable.