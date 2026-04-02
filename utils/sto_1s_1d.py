from qiskit import QuantumCircuit
import numpy as np
from scipy.linalg import null_space

from utils.arithmetic import ArithmeticOperator
from utils.matrix_product_states import Mps


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

    def get_sto_1s_1d_derivative_carthesian(self, qubit_count: int, decay_constant: float, center_offset: int) -> QuantumCircuit:
        
        sto_derivative_circuit = self._sto_1s_1d_derivative_cartesian(qubit_count, decay_constant)

        arithmeticOperator = ArithmeticOperator(sto_derivative_circuit, self.allow_measurement, self.optimize_t_gates)
        qc = arithmeticOperator.add_constant(qubit_count, center_offset)

        return qc
    
    def get_sto_1s_1d_derivative_carthesian_dagger(self, qubit_count: int, decay_constant: float, center_offset: int) -> QuantumCircuit:

        qc = QuantumCircuit(qubit_count)
        arithmeticOperator = ArithmeticOperator(qc, self.allow_measurement, self.optimize_t_gates)
        add_dagger_circuit = arithmeticOperator.subtract_constant(qubit_count, center_offset)

        sto_derivative_dagger_circuit =  self._sto_1s_1d_derivative_cartesian_dagger(qubit_count, decay_constant)

        n_total = add_dagger_circuit.num_qubits  # data + ancilla
        combined = QuantumCircuit(n_total)
        for creg in add_dagger_circuit.cregs:
            combined.add_register(creg)
        combined.compose(add_dagger_circuit, qubits=range(n_total), clbits=list(combined.clbits), inplace=True)
        combined.compose(sto_derivative_dagger_circuit, qubits=range(qubit_count), inplace=True)

        return combined    

    def get_sto_1s_spherical(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        # Decaying exponential state: |ψ⟩ = N · Σ e^(-a·|i|) |i⟩
        return self._sto_1s_spherical(qubit_count, decay_constant)
    
    def get_sto_1s_spherical_dagger(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        return self.get_sto_1s_spherical(qubit_count, decay_constant).inverse()

    def get_sto_1s_1d_potential_cartesian(self, qubit_count: int, decay_constant: float, center_offset: float, potential_offset: float, max_range: int) -> QuantumCircuit:
        
        def potential_times_s1(x):
            s1 = np.exp(-decay_constant * np.abs(x - center_offset))
            V = 1 / np.abs(x - potential_offset)
            f = V * s1
            return f
        
        mps = Mps(potential_times_s1)
        qc = mps.generate_mps_circuit(qubit_count, max_range)
        return qc



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
    
    @staticmethod
    def _sto_1s_1d_derivative_cartesian(qubit_count: int, alpha: float) -> QuantumCircuit:
        """Returns a circuit that prepares the state |ψ⟩ = N · Σ e^(-alpha·|i|) |i⟩."""
        qc = QuantumCircuit(qubit_count)
        last_qubit = qubit_count - 1
        qc.x(last_qubit)

        sto1s = Sto1S._sto_1s_1d_cartesian(qubit_count, alpha)
        qc.compose(sto1s, qubits=range(qubit_count), inplace=True)

        return qc
    
    @staticmethod
    def _sto_1s_1d_derivative_cartesian_dagger(qubit_count: int, alpha: float) -> QuantumCircuit:
        last_qubit = qubit_count - 1
        qc = Sto1S._sto_1s_1d_cartesian_dagger(qubit_count, alpha)
        qc.x(last_qubit)
        return qc