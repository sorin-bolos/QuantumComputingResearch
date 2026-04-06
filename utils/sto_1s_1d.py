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
        
        mps = Mps()
        qc = mps.generate_mps_circuit_from_function(potential_times_s1, qubit_count, max_range)
        return qc
    
    def get_sto_1s_1d_potential_cartesian_dagger(self, qubit_count: int, decay_constant: float, center_offset: float, potential_offset: float, max_range: int) -> QuantumCircuit:
        return self.get_sto_1s_1d_potential_cartesian(qubit_count, decay_constant, center_offset, potential_offset, max_range).inverse()
    
    def get_sto_1s_spherical_jacobian(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        mps = Mps()
        tensors = Sto1S._get_analytical_1s_jacobian_mps_tensors(qubit_count, decay_constant)
        return mps.generate_mps_circuit_from_tensors(tensors, qubit_count)

    def get_sto_1s_spherical_jacobian_dagger(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        return self.get_sto_1s_spherical_jacobian(qubit_count, decay_constant).inverse()   

    @staticmethod
    def _sto_1s_spherical(qubit_count: int, alpha: float) -> QuantumCircuit:
        """Returns a circuit that prepares the state |ψ⟩ = N · Σ e^(-alpha·|i|) |i⟩."""
        qc = QuantumCircuit(qubit_count)

        b = alpha
        for i in range(qubit_count):
            theta = 2 * np.arctan(np.exp(-b))
            qc.ry(theta, i)
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
    
    @staticmethod
    def _norm_sq_xe_neg_ax(qubit_count: int, alpha: float) -> float:
        """Compute sum_{x=0}^{2^n-1} (x * exp(-alpha*x))^2  in O(n)
        via MPS transfer matrix contraction.

        The transfer matrix at site k is  T_k = sum_s  A^s (x) A^s  (Kronecker product)
        with A^0 = I_2 and A^1 = e_k * [[1, p_k],[0, 1]].

        T_k is 4x4, upper-triangular, allowing the full product in O(n).
        """
        # v_L = (1,0) x (1,0)  ->  (1,0,0,0)
        w = np.array([1.0, 0.0, 0.0, 0.0])

        for k in range(qubit_count):
            p = 2.0 ** (qubit_count - 1 - k)
            e2 = np.exp(-2 * alpha * p)     # e_k^2 = exp(-2*alpha*2^{qubit_count-1-k})

            # T_k = I_4  +  e_k^2 * (M_k kron M_k)
            # where M_k = [[1, p], [0, 1]]
            T = np.eye(4)
            T += e2 * np.array([
                [1,  p,  p,  p * p],
                [0,  1,  0,      p],
                [0,  0,  1,      p],
                [0,  0,  0,      1],
            ])
            w = w @ T

        # v_R = (0,1) x (0,1)  ->  (0,0,0,1)
        return w[3]

    @staticmethod
    def _get_analytical_1s_jacobian_mps_tensors(qubit_count: int, alpha: float):
        """Construct MPS tensors for |psi> = (1/N) sum_x x*exp(-alpha*x) |x> analytically.

        Bond dimension is exactly 2 for any n and any alpha > 0.
        Cost: O(n) — no exponential state vector needed.

        Transfer matrices:
        A^{[k]0} = I_2                          (bit k is 0)
        A^{[k]1} = e_k * [[1, w_k], [0, 1]]    (bit k is 1)

        where e_k = exp(-alpha * 2^(n-1-k)) and w_k = 2^(n-1-k) / N.
        """
        norm_sq = Sto1S._norm_sq_xe_neg_ax(qubit_count, alpha)
        norm = np.sqrt(norm_sq)

        if qubit_count == 1:
            e0 = np.exp(-alpha)
            A = np.zeros((1, 2, 1))
            A[0, 1, 0] = e0 / norm
            return [A]

        tensors = []
        for k in range(qubit_count):
            p_k = 2.0 ** (qubit_count - 1 - k)
            e_k = np.exp(-alpha * p_k)
            w_k = p_k / norm

            if k == 0:
                # Left boundary: absorb (1, 0) row vector
                A = np.zeros((1, 2, 2))
                A[0, 0, :] = [1.0, 0.0]
                A[0, 1, :] = [e_k, e_k * w_k]
            elif k == qubit_count - 1:
                # Right boundary: absorb (0, 1)^T column vector
                A = np.zeros((2, 2, 1))
                A[:, 0, 0] = [0.0, 1.0]
                A[:, 1, 0] = [e_k * w_k, e_k]
            else:
                # Interior: e_k * upper-triangular for s=1, identity for s=0
                A = np.zeros((2, 2, 2))
                A[:, 0, :] = np.eye(2)
                A[:, 1, :] = [[e_k, e_k * w_k],
                            [0.0,        e_k]]

            tensors.append(A)
        return tensors