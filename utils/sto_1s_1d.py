from qiskit import QuantumCircuit
import numpy as np

from utils.arithmetic import ArithmeticOperator


class Sto1S:
    """Implements the STO-1s-1d operation."""
    def __init__(self):
        pass

    def get_sto_1s_1d_carthesian(self, qubit_count: int, decay_constant: float, max_range: float, center_offset: int) -> QuantumCircuit:
        # Decaying exponential state: |ψ⟩ = N · Σ e^(-a·|i|) |i⟩

        scale = max_range / (2 ** qubit_count)
        scaled_constant = decay_constant * scale

        sto_circuit = self._sto_1s_1d_cartesian(qubit_count, scaled_constant)

        arithmeticOperator = ArithmeticOperator(sto_circuit)
        qc = arithmeticOperator.add_constant(qubit_count, center_offset)

        return qc
    
    def get_sto_1s_1d_carthesian_dagger(self, qubit_count: int, decay_constant: float, max_range: float, center_offset: int) -> QuantumCircuit:
        return self.get_sto_1s_1d_carthesian(qubit_count, decay_constant, max_range, center_offset).inverse()

    def get_sto_1s_spherical(self, qubit_count: int, decay_constant: float, max_range: float) -> QuantumCircuit:
        # Decaying exponential state: |ψ⟩ = N · Σ e^(-a·|i|) |i⟩

        scale = max_range / (2 ** qubit_count)
        scaled_constant = decay_constant * scale

        return self._sto_1s_spherical(qubit_count, scaled_constant)
    
    def get_sto_1s_spherical_dagger(self, qubit_count: int, decay_constant: float, max_range: float) -> QuantumCircuit:
        return self.get_sto_1s_spherical(qubit_count, decay_constant, max_range).inverse()

    
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