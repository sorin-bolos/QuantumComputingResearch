from qiskit import QuantumCircuit
import numpy as np

from utils.matrix_product_states import Mps

class Sto2S:
    def __init__(self, allow_measurement: bool = True, optimize_t_gates: bool =True):
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates

    def get_sto_2s_spherical(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        mps = Mps()
        tensors = Sto2S._analytical_mps_2s_orbital_mps_tensors(qubit_count, decay_constant)
        return mps.generate_mps_circuit_from_tensors(tensors, qubit_count)
    
    def get_sto_2s_spherical_dagger(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        return self.get_sto_2s_spherical(qubit_count, decay_constant).inverse()
    
    def get_sto_2s_spherical_jacobian(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        mps = Mps()
        tensors = Sto2S._analytical_mps_2s_jacobian_mps_tensors(qubit_count, decay_constant)
        return mps.generate_mps_circuit_from_tensors(tensors, qubit_count)

    def get_sto_2s_spherical_jacobian_dagger(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        return self.get_sto_2s_spherical_jacobian(qubit_count, decay_constant).inverse()

    
    @staticmethod
    def _norm_sq_2s_orbital(n, alpha):
        """Compute sum_{x=0}^{2^n-1} ((2 - ax) * exp(-alpha*x/2))^2 in O(n)
        via 9x9 transfer matrix contraction (chi=3, so chi^2=9).
        """
        l = np.array([2.0, 1.0, 0.0])
        r = np.array([1.0, 0.0, 1.0])

        v = np.kron(l, l)  # 9-dim left boundary vector

        for k in range(n):
            p_k = 2.0 ** (n - 1 - k)
            e_k = np.exp(-p_k * alpha / 2.0)

            N_k = e_k * np.array([
                [1.0, 0.0,        0.0],
                [0.0, 1.0, -p_k * alpha],
                [0.0, 0.0,        1.0],
            ])

            T_k = np.eye(9) + np.kron(N_k, N_k)
            v = v @ T_k

        return v @ np.kron(r, r)
    
    @staticmethod
    def _analytical_mps_2s_orbital_mps_tensors(n, alpha):
        """Construct MPS tensors for |psi> = (1/N) sum_x (2-alpha*x)*exp(-alpha*x/2) |x>.

        Bond dimension is exactly 3 for any n.
        Cost: O(n) — no exponential state vector needed.

        Transfer matrices (unnormalized):
            A^{[k]0} = I_3
            A^{[k]1} = e_k * [[1, 0, 0], [0, 1, -p_k*alpha], [0, 0, 1]]

        Left boundary: (2, 1, 0),  Right boundary: (1, 0, 1)^T
        Normalization absorbed into the left boundary.
        """
        norm = np.sqrt(Sto2S._norm_sq_2s_orbital(n, alpha))

        if n == 1:
            e0 = np.exp(-1.0 * alpha / 2.0)
            A = np.zeros((1, 2, 1))
            A[0, 0, 0] = 2.0 / norm
            A[0, 1, 0] = (2.0 - 1.0 / alpha) * e0 / norm
            return [A]

        tensors = []
        for k in range(n):
            p_k = 2.0 ** (n - 1 - k)
            e_k = np.exp(-p_k * alpha / 2.0)

            if k == 0:
                # Left boundary: absorb (2/N, 1/N, 0)
                A = np.zeros((1, 2, 3))
                A[0, 0, :] = [2.0 / norm, 1.0 / norm, 0.0]
                A[0, 1, :] = [2.0 * e_k / norm,
                            e_k / norm,
                            -e_k * p_k * alpha / norm]
            elif k == n - 1:
                # Right boundary: absorb (1, 0, 1)^T
                A = np.zeros((3, 2, 1))
                A[:, 0, 0] = [1.0, 0.0, 1.0]
                A[:, 1, 0] = [e_k, -e_k * p_k * alpha, e_k]
            else:
                # Interior: block-diagonal with off-diagonal coupling
                A = np.zeros((3, 2, 3))
                A[:, 0, :] = np.eye(3)
                A[:, 1, :] = [[e_k,  0.0,              0.0],
                            [0.0,  e_k, -e_k * p_k * alpha],
                            [0.0,  0.0,              e_k]]

            tensors.append(A)
        return tensors

    @staticmethod
    def _norm_sq_2s_jacobian(n, alpha):
        """Compute sum_{r=0}^{2^n-1} (r*(2-alpha*r)*exp(-alpha*r/2))^2  in O(n)
        via 9x9 transfer matrix contraction (chi=3, so chi^2=9).

        Uses the quadratic running-sum transfer matrices:
            A^0 = I_3,  A^1 = e_k * [[1, p_k, p_k^2], [0, 1, 2*p_k], [0, 0, 1]]
        with left boundary (1, 0, 0) and right boundary (0, 2, -alpha).
        """
        l = np.array([1.0, 0.0, 0.0])
        r = np.array([0.0, 2.0, -1.0 * alpha])

        v = np.kron(l, l)  # 9-dim left boundary vector

        for k in range(n):
            p_k = 2.0 ** (n - 1 - k)
            e_k = np.exp(-p_k * alpha / 2.0)

            N_k = e_k * np.array([
                [1.0,  p_k,  p_k * p_k],
                [0.0,  1.0,  2.0 * p_k],
                [0.0,  0.0,        1.0],
            ])

            T_k = np.eye(9) + np.kron(N_k, N_k)
            v = v @ T_k

        return v @ np.kron(r, r)
    
    @staticmethod
    def _analytical_mps_2s_jacobian_mps_tensors(n, alpha):
        """Construct MPS tensors for |psi> = (1/N) sum_r r*(2-alpha*r)*exp(-alpha*r/2) |r>.

        Bond dimension is exactly 3 for any n.
        Cost: O(n) — no exponential state vector needed.

        The three bond states track the quadratic running sum:
            state 0: constant (1)
            state 1: linear sum x = sum_j s_j p_j
            state 2: quadratic sum x^2

        Transfer matrices:
            A^{[k]0} = I_3
            A^{[k]1} = e_k * [[1, p_k, p_k^2], [0, 1, 2*p_k], [0, 0, 1]]

        Left boundary: (1, 0, 0) / N
        Right boundary: (0, 2, -alpha)^T
        """
        norm = np.sqrt(Sto2S._norm_sq_2s_jacobian(n, alpha))

        if n == 1:
            e0 = np.exp(-1.0 * alpha / 2.0)
            A = np.zeros((1, 2, 1))
            A[0, 0, 0] = 0.0  # f(0) = 0
            A[0, 1, 0] = (2.0 - alpha) * e0 / norm  # f(1) = 1*(2-1*alpha)*e^{-1*(alpha)/2}
            return [A]

        tensors = []
        for k in range(n):
            p_k = 2.0 ** (n - 1 - k)
            e_k = np.exp(-p_k * alpha / 2.0)

            if k == 0:
                # Left boundary: absorb (1/N, 0, 0)
                A = np.zeros((1, 2, 3))
                A[0, 0, :] = [1.0 / norm, 0.0, 0.0]
                A[0, 1, :] = [e_k / norm,
                            e_k * p_k / norm,
                            e_k * p_k * p_k / norm]
            elif k == n - 1:
                # Right boundary: absorb (0, 2, -1/alpha)^T
                A = np.zeros((3, 2, 1))
                A[:, 0, 0] = [0.0, 2.0, -1.0 * alpha]
                A[:, 1, 0] = [e_k * (2.0 * p_k - p_k * p_k * alpha),
                            e_k * (2.0 - 2.0 * p_k * alpha),
                            -e_k * alpha]
            else:
                # Interior: upper-triangular for s=1, identity for s=0
                A = np.zeros((3, 2, 3))
                A[:, 0, :] = np.eye(3)
                A[:, 1, :] = [[e_k, e_k * p_k,      e_k * p_k * p_k],
                            [0.0,       e_k,      e_k * 2.0 * p_k],
                            [0.0,       0.0,                  e_k]]

            tensors.append(A)
        return tensors