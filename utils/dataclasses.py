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
        print(f"  Center distance (discretized) : {self.used_center_distance}")
        print(f"  Scaled distance (in grid) : {self.scaled_center_distance}")
        print(f"  Exact result             : {self.exact_result:.6f}")


@dataclass
class Results:
    context: IntegralContext

    analitical_zero_amplitude: float
    analitical_zero_probablity: float
    errors_for_analitical: Errors

    sampled_zero_amplitude: float = None
    sampled_zero_probability: float = None
    errors_for_sampled: Errors = None

    noisy_sampled_zero_amplitude: float = None
    noisy_sampled_zero_probability: float = None
    errors_for_noisy_sampled: Errors = None

    estimator_zero_amplitude: Optional[float] = None
    estimator_zero_probability: Optional[float] = None
    errors_for_estimator: Optional[Errors] = None

    ibm_sampler_zero_amplitude: Optional[float] = None
    ibm_sampler_zero_probability: Optional[float] = None
    errors_for_ibm_sampler: Optional[Errors] = None

    ibm_estimator_zero_amplitude: Optional[float] = None
    ibm_estimator_zero_probability: Optional[float] = None
    errors_for_ibm_estimator: Optional[Errors] = None

    noisy_simulation_stats: Optional[CircuitStats] = None
    ibm_backend_stats: Optional[CircuitStats] = None

    def print(self):
        self.context.print()

        self.noisy_simulation_stats.print()
        self.ibm_backend_stats.print()

        print()
        print("  ── Analytical (noiseless statevector) ──")
        print(f"  Amplitude    : {self.analitical_zero_amplitude:.6f}")
        print(f"  Probability  : {self.analitical_zero_probablity:.6f}")
        if self.errors_for_analitical is not None:
            self.errors_for_analitical.print()

        if self.sampled_zero_amplitude is not None:
            print()
            print("  ── Sampled (noiseless shot-based) ──")
            print(f"  Amplitude    : {self.sampled_zero_amplitude:.6f}")
            print(f"  Probability  : {self.sampled_zero_probability:.6f}")
            if self.errors_for_sampled is not None:
                self.errors_for_sampled.print()

        if self.noisy_sampled_zero_amplitude is not None:
            print()
            print("  ── Noisy sampler (fake backend) ──")
            print(f"  Amplitude    : {self.noisy_sampled_zero_amplitude:.6f}")
            print(f"  Probability  : {self.noisy_sampled_zero_probability:.6f}")
            if self.errors_for_noisy_sampled is not None:
                self.errors_for_noisy_sampled.print()

        if self.estimator_zero_amplitude is not None:
            print()
            print("  ── Noisy estimator + ZNE (fake backend) ──")
            print(f"  Amplitude    : {self.estimator_zero_amplitude:.6f}")
            print(f"  Probability  : {self.estimator_zero_probability:.6f}")
            if self.errors_for_estimator is not None:
                self.errors_for_estimator.print()

        if self.ibm_sampler_zero_amplitude is not None:
            print()
            print("  ── IBM hardware sampler ──")
            print(f"  Amplitude    : {self.ibm_sampler_zero_amplitude:.6f}")
            print(f"  Probability  : {self.ibm_sampler_zero_probability:.6f}")
            if self.errors_for_ibm_sampler is not None:
                self.errors_for_ibm_sampler.print()

        if self.ibm_estimator_zero_amplitude is not None:
            print()
            print("  ── IBM hardware estimator + ZNE ──")
            print(f"  Amplitude    : {self.ibm_estimator_zero_amplitude:.6f}")
            print(f"  Probability  : {self.ibm_estimator_zero_probability:.6f}")
            if self.errors_for_ibm_estimator is not None:
                self.errors_for_ibm_estimator.print()
