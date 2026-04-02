from dataclasses import dataclass
from typing import Optional

__all__ = [
    "CircuitStats",
    "Errors",
    "IntegralContext",
    "Results",
]


@dataclass
class CircuitStats:
    """Statistics for a circuit transpiled to an IBM native gate set."""
    backend_name: str
    num_qubits: int
    depth: int
    single_qubit_gates: int   # rz + sx + x
    two_qubit_gates: int      # ecr
    t_gates_logical: int      # t + tdg in the original (pre-transpile) circuit
    gate_counts: dict

    def print(self):
        print()
        print(f"  Backend name         : {self.backend_name}")
        print(f"  Qubits (transpiled)  : {self.num_qubits}")
        print(f"  Depth                : {self.depth}")
        print(f"  Single-qubit gates   : {self.single_qubit_gates}  (rz + sx + x)")
        print(f"  Two-qubit gates (ecr): {self.two_qubit_gates}")
        print(f"  T gates (logical)    : {self.t_gates_logical}")
        print(f"  All gate counts      : {self.gate_counts}")


@dataclass
class Errors:
    error_vs_continuous: float
    percent_error_vs_continuous: float
    discretisation_error: float
    percent_discretisation_error: float
    shot_noise: float
    percent_shot_noise: float

    def print(self):
        print(f"  Error vs continuous  : {self.error_vs_continuous:.6f}  ({self.percent_error_vs_continuous:.2f} %)")
        print(f"  Discretisation error : {self.discretisation_error:.6f}  ({self.percent_discretisation_error:.2f} %)")
        print(f"  Shot noise           : {self.shot_noise:.6f}  ({self.percent_shot_noise:.2f} %)")


@dataclass
class IntegralContext:
    used_center_distance: float
    scaled_center_distance: int
    exact_result: float

    def print(self):
        print()
        print(f"  Center distance (discretized) : {self.used_center_distance}")
        print(f"  Scaled distance (in grid) : {self.scaled_center_distance}")
        print(f"  Exact result             : {self.exact_result:.6f}")

@dataclass
class RunResults:
    run_name: str
    run_result: float

    def print(self):
        print()
        print(f"  {self.run_name} result : {self.run_result:.6f}")

@dataclass
class SimulationResults:
    name: str
    result: float
    errors: Errors

    def print(self):
        print()
        print(f"  {self.name} result : {self.result:.6f}")
        self.errors.print()


@dataclass
class Results:
    context: IntegralContext
    stats: list[CircuitStats]
    results: list[RunResults]

    def print(self):
        self.context.print()
        for stat in self.stats:
            stat.print()

        print()
        for result in self.results:
            result.print()
