"""Estimator-based executor for real IBM Quantum hardware."""

import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2

__all__ = ["IbmEstimatorExecutor"]


class IbmEstimatorExecutor:
    """Computes zero-state probabilities on real IBM Quantum hardware.

    Uses ``EstimatorV2`` via ``qiskit_ibm_runtime``.  All error suppression
    and mitigation options are fully active on real hardware.

    The quantity computed is P(|0…0⟩) = ⟨ψ|(|0⟩⟨0|)^⊗n|ψ⟩, where |ψ⟩ is
    the state prepared by the circuit and n = data_qubit_count.

    Parameters
    ----------
    backend : IBM backend
        A backend instance from ``QiskitRuntimeService``, e.g.::

            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backend = service.backend("ibm_sherbrooke")
            backend = service.least_busy(operational=True, simulator=False)

    optimization_level : int, optional
        Transpiler optimisation level 0–3 (default 3).
    enable_dd : bool, optional
        Enable Dynamical Decoupling (XY4 by default).
        Incompatible with dynamic circuits (``if_else`` / mid-circuit
        measurements).  Default False.
    dd_sequence : str, optional
        DD sequence: ``'XY4'`` (default), ``'XX'``, ``'XpXm'``.
    enable_twirling : bool, optional
        Enable Pauli twirling on gates and measurements.  Default False.
    twirling_num_randomizations : int, optional
        Number of random twirl circuits (default 32).
    enable_measure_mitigation : bool, optional
        Enable TREX readout error mitigation (resilience level 1).
        Default False.
    enable_zne : bool, optional
        Enable Zero Noise Extrapolation (resilience level 2).
        Runs the circuit at multiple noise amplification levels and
        extrapolates the expectation value to zero noise.  Default False.
    zne_noise_factors : list of int, optional
        Noise amplification factors (default ``[1, 3, 5]``).  Must be odd
        integers.  More factors improve extrapolation accuracy at the cost of
        additional circuit executions.
    zne_extrapolator : str or list of str, optional
        Extrapolation method: ``'exponential'`` (default), ``'linear'``,
        ``'polynomial_degree_N'``, ``'richardson'``.  Pass a list to let the
        runtime select the best fit automatically.

    Notes
    -----
    Credentials must be saved before use::

        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(channel="ibm_quantum", token="MY_TOKEN")

    ``get_probability_of_zero`` and ``get_amplitude_of_zero`` block until the
    IBM Quantum job completes.
    """

    def __init__(
        self,
        backend,
        optimization_level: int = 3,
        enable_dd: bool = False,
        dd_sequence: str = 'XY4',
        enable_twirling: bool = False,
        twirling_num_randomizations: int = 32,
        enable_measure_mitigation: bool = False,
        enable_zne: bool = False,
        zne_noise_factors: list = None,
        zne_extrapolator=None,
    ):
        self.backend = backend
        self.optimization_level = optimization_level
        self.pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=self.backend,
        )
        self.estimator = EstimatorV2(mode=self.backend)

        if enable_dd:
            self.estimator.options.dynamical_decoupling.enable = True
            self.estimator.options.dynamical_decoupling.sequence_type = dd_sequence

        if enable_twirling:
            self.estimator.options.twirling.enable_gates = True
            self.estimator.options.twirling.enable_measure = True
            self.estimator.options.twirling.num_randomizations = twirling_num_randomizations

        if enable_measure_mitigation:
            self.estimator.options.resilience.measure_mitigation = True

        if enable_zne:
            self.estimator.options.resilience.zne_mitigation = True
            self.estimator.options.resilience.zne.noise_factors = (
                zne_noise_factors if zne_noise_factors is not None else [1, 3, 5]
            )
            self.estimator.options.resilience.zne.extrapolator = (
                zne_extrapolator if zne_extrapolator is not None else 'exponential'
            )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _zero_state_projector(n_data: int, n_total: int) -> SparsePauliOp:
        """Build O = |0…0⟩⟨0…0| ⊗ I^anc as a SparsePauliOp.

        In Qiskit's string convention the rightmost character is qubit 0.
        Data qubits are 0…n_data-1 (rightmost); ancilla are n_data…n_total-1
        (leftmost, all identity).
        """
        anc_prefix = 'I' * (n_total - n_data)
        coeff = 1.0 / 2 ** n_data
        terms = [
            (anc_prefix + ''.join(combo), coeff)
            for combo in itertools.product('IZ', repeat=n_data)
        ]
        return SparsePauliOp.from_list(terms)

    def _transpile(self, qc: QuantumCircuit) -> QuantumCircuit:
        return self.pm.run(qc)

    # ── public API ────────────────────────────────────────────────────────────

    def get_probability_of_zero(
        self, qc: QuantumCircuit, data_qubit_count: int, shots: int = 1024
    ) -> float:
        """Estimate P(|0…0⟩) on *data_qubit_count* qubits via EstimatorV2.

        Parameters
        ----------
        qc : QuantumCircuit
            Circuit preparing the state |ψ⟩.  Must not contain a final
            measurement (EstimatorV2 handles measurement internally).
        data_qubit_count : int
            Number of data qubits (qubits 0…data_qubit_count-1).
        shots : int, optional
            Number of shots per circuit execution (default 1024).
            When ZNE is enabled the total shots are multiplied by the number
            of noise factors.

        Returns
        -------
        float
            Estimated P(|0…0⟩), clipped to [0, 1].
            ZNE extrapolation can produce slightly negative values; clipping
            corrects for this.
        """
        obs = self._zero_state_projector(data_qubit_count, qc.num_qubits)
        qc_isa = self._transpile(qc)
        obs_isa = obs.apply_layout(qc_isa.layout)

        self.estimator.options.default_shots = shots
        result = self.estimator.run([(qc_isa, obs_isa)]).result()
        return float(np.clip(result[0].data.evs, 0.0, 1.0))

    def get_amplitude_of_zero(
        self, qc: QuantumCircuit, data_qubit_count: int, shots: int = 1024
    ) -> float:
        """Estimate √P(|0…0⟩) = |⟨0…0|ψ⟩| via EstimatorV2.

        Parameters
        ----------
        qc : QuantumCircuit
        data_qubit_count : int
        shots : int, optional

        Returns
        -------
        float
            √P(|0…0⟩).
        """
        return float(np.sqrt(self.get_probability_of_zero(qc, data_qubit_count, shots)))
