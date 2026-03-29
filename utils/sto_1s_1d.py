from qiskit import QuantumCircuit
import numpy as np

from utils.arithmetic import ArithmeticOperator


class Sto1S:
    """Implements the STO-1s-1d operation."""
    def __init__(self, allow_measurement: bool = True, optimize_t_gates: bool =True):
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates

    def get_sto_1s_1d_carthesian(self, qubit_count: int, decay_constant: float, center_offset: int) -> QuantumCircuit:
        # Decaying exponential state: |ψ⟩ = N · Σ e^(-a·|i|) |i⟩

        sto_circuit = self._sto_1s_1d_cartesian(qubit_count, decay_constant)

        arithmeticOperator = ArithmeticOperator(sto_circuit, self.allow_measurement, self.optimize_t_gates)
        qc = arithmeticOperator.add_constant(qubit_count, center_offset)

        return qc
    
    def get_sto_1s_1d_carthesian_dagger(self, qubit_count: int, decay_constant: float, center_offset: int) -> QuantumCircuit:

        qc = QuantumCircuit(qubit_count)
        arithmeticOperator = ArithmeticOperator(qc, self.allow_measurement, self.optimize_t_gates)
        add_dagger_circuit = arithmeticOperator.subtract_constant(qubit_count, center_offset)

        sto_circuit_dagger_circuit =  self._sto_1s_1d_cartesian_dagger(qubit_count, decay_constant)

        n_total = add_dagger_circuit.num_qubits  # data + ancilla
        combined = QuantumCircuit(n_total)
        for creg in add_dagger_circuit.cregs:
            combined.add_register(creg)
        combined.compose(add_dagger_circuit, qubits=range(n_total), clbits=list(combined.clbits), inplace=True)
        combined.compose(sto_circuit_dagger_circuit, qubits=range(qubit_count), inplace=True)

        return combined

        

    def get_sto_1s_spherical(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        # Decaying exponential state: |ψ⟩ = N · Σ e^(-a·|i|) |i⟩
        return self._sto_1s_spherical(qubit_count, decay_constant)
    
    def get_sto_1s_spherical_dagger(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        return self.get_sto_1s_spherical(qubit_count, decay_constant).inverse()

    
    @staticmethod
    def _sto_1s_spherical(qubit_count: int, alpha: float) -> QuantumCircuit:
        """Returns a circuit that prepares the state |ψ⟩ = N · Σ e^(-alpha·|i|) |i⟩."""
        qc = QuantumCircuit(qubit_count)
        last_qubit = qubit_count - 1

        b = alpha
        for i in range(qubit_count-1):
            theta = 2 * np.arctan(np.exp(-b))
            qc.cry(theta, last_qubit, i)
            b = b * 2

        return qc

    @staticmethod
    def _sto_1s_1d_cartesian(qubit_count: int, alpha: float) -> QuantumCircuit:
        """Returns a circuit that prepares the state |ψ⟩ = N · Σ e^(-alpha·|i|) |i⟩."""
        qc = QuantumCircuit(qubit_count)
        last_qubit = qubit_count - 1

        qc.h(last_qubit)

        b = alpha
        for i in range(qubit_count-1):
            theta = 2 * np.arctan(np.exp(b))
            qc.cry(theta, last_qubit, i)
            b = b * 2
        qc.x(last_qubit)

        b = alpha
        for i in range(qubit_count-1):
            theta = 2 * np.arctan(np.exp(-b))
            qc.cry(theta, last_qubit, i)
            b = b * 2

        qc.x(last_qubit)

        return qc
    
    @staticmethod
    def _sto_1s_1d_cartesian_dagger(qubit_count: int, alpha: float) -> QuantumCircuit:
        """Returns the adjoint of prepare_sto_state."""
        qc = QuantumCircuit(qubit_count)
        last_qubit = qubit_count - 1

        qc.x(last_qubit)

        b = alpha
        for i in range(qubit_count-1):
            theta = 2 * np.arctan(np.exp(-b))
            qc.cry(-theta, last_qubit, i)
            b = b * 2
            
        qc.x(last_qubit)

        b = alpha
        for i in range(qubit_count-1):
            theta = 2 * np.arctan(np.exp(b))
            qc.cry(-theta, last_qubit, i)
            b = b * 2

        qc.h(last_qubit)
        return qc