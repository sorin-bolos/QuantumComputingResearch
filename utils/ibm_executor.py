"""IBM Quantum hardware executor — real-device counterpart to NoisySimulationExecutor."""

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2

from utils.dataclasses import CircuitStats

__all__ = ["IbmExecutor"]


class IbmExecutor:
    """Executes quantum circuits on real IBM Quantum hardware.

    Drop-in replacement for ``NoisySimulationExecutor`` with identical public
    methods.  Uses ``SamplerV2`` via ``qiskit_ibm_runtime`` against a real
    backend obtained from ``QiskitRuntimeService``.

    Parameters
    ----------
    backend : IBM backend
        A backend instance from ``QiskitRuntimeService``, e.g.::

            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backend = service.backend("ibm_sherbrooke")
            # or pick least-busy device:
            backend = service.least_busy(operational=True, simulator=False)

    optimization_level : int, optional
        Transpiler optimisation level 0–3 (default 3).
    enable_dd : bool, optional
        Enable Dynamical Decoupling (XY4 by default) for error suppression.
        **Incompatible with dynamic circuits** (circuits containing ``if_else`` /
        mid-circuit measurements).  Default False.
    dd_sequence : str, optional
        DD sequence type: ``'XY4'`` (default), ``'XX'``, ``'XpXm'``.
    enable_twirling : bool, optional
        Enable Pauli twirling on gates and measurements.  Fully active on real
        hardware (unlike fake backends).  Default False.
    twirling_num_randomizations : int, optional
        Number of random twirl circuits to average over (default 32).
    enable_m3 : bool, optional
        Enable M3 readout error correction.  Calibrates against the backend at
        construction time; recalibrate periodically on real hardware as error
        rates drift (every few hours).  Requires ``pip install mthree``.
        Default False.

    Notes
    -----
    Credentials must be saved before use::

        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(channel="ibm_quantum", token="MY_TOKEN")

    Jobs are submitted to the IBM Quantum queue and may wait before running.
    ``sample_*`` calls block until the job completes.
    """

    def __init__(
        self,
        backend,
        optimization_level: int = 3,
        enable_dd: bool = False,
        dd_sequence: str = 'XY4',
        enable_twirling: bool = False,
        twirling_num_randomizations: int = 32,
        enable_m3: bool = False,
    ):
        self.backend = backend
        self.optimization_level = optimization_level
        self.pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=self.backend,
        )
        self.sampler = SamplerV2(mode=self.backend)

        if enable_dd:
            self.sampler.options.dynamical_decoupling.enable = True
            self.sampler.options.dynamical_decoupling.sequence_type = dd_sequence

        if enable_twirling:
            self.sampler.options.twirling.enable_gates = True
            self.sampler.options.twirling.enable_measure = True
            self.sampler.options.twirling.num_randomizations = twirling_num_randomizations

        self._m3 = None
        if enable_m3:
            self._init_m3()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _init_m3(self) -> None:
        try:
            import mthree
        except ImportError:
            raise ImportError(
                "M3 mitigation requires the mthree package: pip install mthree"
            )
        self._m3 = mthree.M3Mitigation(self.backend)
        self._m3.cals_from_system(list(range(self.backend.num_qubits)))

    def _physical_qubits_for_register(self, qc_isa: QuantumCircuit, reg_name: str) -> list:
        """Return physical qubit indices in classical-register bit order."""
        reg = next(r for r in qc_isa.cregs if r.name == reg_name)
        bit_to_pos = {bit: i for i, bit in enumerate(reg)}
        pos_to_phys: dict = {}
        for instruction in qc_isa.data:
            if instruction.operation.name == 'measure':
                clbit = instruction.clbits[0]
                if clbit in bit_to_pos:
                    phys = qc_isa.find_bit(instruction.qubits[0]).index
                    pos_to_phys[bit_to_pos[clbit]] = phys
        return [pos_to_phys[i] for i in range(len(reg))]

    def _transpile(self, qc: QuantumCircuit) -> QuantumCircuit:
        return self.pm.run(qc)

    def _run(self, qc_isa: QuantumCircuit, reg_name: str, shots: int) -> dict:
        result = self.sampler.run([qc_isa], shots=shots).result()
        counts = getattr(result[0].data, reg_name).get_counts()

        if self._m3 is not None:
            qubits = self._physical_qubits_for_register(qc_isa, reg_name)
            quasi = self._m3.apply_correction(counts, qubits)
            probs = quasi.nearest_probability_distribution()
            counts = {k: round(v * shots) for k, v in probs.items()}

        return counts

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample_measurement_counts(self, qc, data_qubit_count: int, shots: int = 1024) -> dict:
        """Submit *qc* to real hardware and return per-data-qubit counts.

        Only qubits 0 … data_qubit_count-1 are measured.

        Parameters
        ----------
        qc : QuantumCircuit
        data_qubit_count : int
        shots : int, optional

        Returns
        -------
        dict
            Bitstring → count. All 2**data_qubit_count basis states present.
        """
        qc_meas = qc.copy()
        meas_reg = ClassicalRegister(data_qubit_count, 'meas')
        qc_meas.add_register(meas_reg)
        qc_meas.measure(list(range(data_qubit_count)), meas_reg)

        qc_isa = self._transpile(qc_meas)
        counts = self._run(qc_isa, 'meas', shots)

        all_basis = [format(i, f'0{data_qubit_count}b') for i in range(2 ** data_qubit_count)]
        return {b: counts.get(b, 0) for b in all_basis}

    def sample_raw_measurement_counts(self, qc, shots: int = 1024) -> dict:
        """Submit *qc* to real hardware measuring all qubits.

        Parameters
        ----------
        qc : QuantumCircuit
        shots : int, optional

        Returns
        -------
        dict
            Bitstring → count. Only observed states included.
        """
        qc_meas = qc.copy()
        raw_reg = ClassicalRegister(qc.num_qubits, 'raw')
        qc_meas.add_register(raw_reg)
        qc_meas.measure(list(range(qc.num_qubits)), raw_reg)

        qc_isa = self._transpile(qc_meas)
        return self._run(qc_isa, 'raw', shots)

    # ── circuit statistics ────────────────────────────────────────────────────

    def print_circuit_stats(self, qc) -> None:
        """Print a formatted summary of circuit statistics after transpilation.

        Parameters
        ----------
        qc : QuantumCircuit
        """
        s = self.get_circuit_stats(qc)
        print(f"Backend              : {self.backend.name}")
        print(f"Qubits (transpiled)  : {s.num_qubits}")
        print(f"Depth                : {s.depth}")
        print(f"Single-qubit gates   : {s.single_qubit_gates}  (rz + sx + x)")
        print(f"Two-qubit gates (ecr): {s.two_qubit_gates}")
        print(f"T gates (logical)    : {s.t_gates_logical}")
        print(f"All gate counts      : {s.gate_counts}")

    def get_circuit_stats(self, qc) -> CircuitStats:
        """Transpile *qc* to the backend's native gates and return statistics.

        Parameters
        ----------
        qc : QuantumCircuit

        Returns
        -------
        CircuitStats
        """
        logical_ops = qc.count_ops()
        t_gates_logical = logical_ops.get('t', 0) + logical_ops.get('tdg', 0)

        qc_t = self._transpile(qc)
        ops = dict(qc_t.count_ops())

        return CircuitStats(
            num_qubits=qc_t.num_qubits,
            depth=qc_t.depth(),
            single_qubit_gates=ops.get('rz', 0) + ops.get('sx', 0) + ops.get('x', 0),
            two_qubit_gates=ops.get('ecr', 0),
            t_gates_logical=t_gates_logical,
            gate_counts=ops,
        )
