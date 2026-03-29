import warnings

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passmanager import PassManager
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.fake_provider import (
    FakeSherbrooke,   # 127 qubits
    FakeTorino,       # 133 qubits
    FakeBrisbane,     # 127 qubits
)

from utils.dataclasses import CircuitStats

class ResourceEstimator:
    def __init__(self):
        pass

    def _get_pass_manager(
        self,
        backend,
        optimization_level
    ):
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend,
        )
        return pm

    def _transpile(self, qc: QuantumCircuit, pm: PassManager) -> QuantumCircuit:
        return pm.run(qc)
            
    def get_circuit_stats(self, 
                          qc,
                          backend=None,
                          optimization_level: int = 3
                         ) -> CircuitStats:
        """Transpile *qc* to the backend's native gates and return statistics.

        Parameters
        ----------
        qc : QuantumCircuit

        Returns
        -------
        CircuitStats
            num_qubits       — qubits after transpilation
            depth            — circuit depth after transpilation
            single_qubit_gates — rz + sx + x count
            two_qubit_gates  — ecr count
            t_gates_logical  — t + tdg in the original logical circuit
                               (T gates become rz(π/4) after transpilation)
            gate_counts      — full gate count dict after transpilation
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            backend = backend or FakeSherbrooke()

        logical_ops = qc.count_ops()
        t_gates_logical = logical_ops.get('t', 0) + logical_ops.get('tdg', 0)

        pm = self._get_pass_manager(backend, optimization_level)
        qc_t = self._transpile(qc, pm)
        ops = dict(qc_t.count_ops())

        return CircuitStats(
            num_qubits=qc_t.num_qubits,
            depth=qc_t.depth(),
            single_qubit_gates=ops.get('rz', 0) + ops.get('sx', 0) + ops.get('x', 0),
            two_qubit_gates=ops.get('ecr', 0),
            t_gates_logical=t_gates_logical,
            gate_counts=ops,
        )