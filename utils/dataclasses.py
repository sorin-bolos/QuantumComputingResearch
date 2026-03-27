from dataclasses import dataclass

__all__ = [
    "CircuitStats",
    "Results"
]


@dataclass
class CircuitStats:
    """Statistics for a circuit transpiled to an IBM native gate set."""
    num_qubits: int
    depth: int
    single_qubit_gates: int   # rz + sx + x
    two_qubit_gates: int      # ecr
    t_gates_logical: int      # t + tdg in the original (pre-transpile) circuit
    gate_counts: dict

@dataclass
class Results:
    analitical_zero_amplitude: float
    analitical_zero_probablity: float
    sampled_zero_amplitude: float
    sampled_zero_probability: float
    noisy_sampled_zero_amplitude: float
    noisy_sampled_zero_probability: float