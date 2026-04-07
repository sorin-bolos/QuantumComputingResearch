from qiskit import QuantumCircuit
import numpy as np

from utils.matrix_product_states import Mps

class Sto2P:
    def __init__(self, allow_measurement: bool = True, optimize_t_gates: bool = True):
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates

    def get_sto_2p_spherical(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        mps = Mps()
        tensors = Sto2P._analytical_mps_2p_orbital_mps_tensors(qubit_count, decay_constant)
        return mps.generate_mps_circuit_from_tensors(tensors, qubit_count)

    def get_sto_2p_spherical_dagger(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        return self.get_sto_2p_spherical(qubit_count, decay_constant).inverse()

    def get_sto_2p_spherical_jacobian(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        mps = Mps()
        tensors = Sto2P._analytical_mps_2p_jacobian_mps_tensors(qubit_count, decay_constant)
        return mps.generate_mps_circuit_from_tensors(tensors, qubit_count)

    def get_sto_2p_spherical_jacobian_dagger(self, qubit_count: int, decay_constant: float) -> QuantumCircuit:
        return self.get_sto_2p_spherical_jacobian(qubit_count, decay_constant).inverse()

    # ------------------------------------------------------------------ #
    #  2p orbital  —  f(r) = r * exp(-alpha*r/2)                         #
    #  Same structure as 1s Jacobian (chi=2) but e_k = exp(-alpha*p_k/2) #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _norm_sq_2p_orbital(n, alpha):
        """Compute sum_{r=0}^{2^n-1} r^2 * exp(-alpha*r) in O(n)
        via 4x4 transfer matrix contraction (chi=2, so chi^2=4).

        Transfer matrix at site k:
            A^0 = I_2,  A^1 = e_k * [[1, p_k], [0, 1]]
        with e_k = exp(-alpha*p_k/2), so e_k^2 = exp(-alpha*p_k).
        Left boundary (1,0), right boundary (0,1).
        """
        w = np.array([1.0, 0.0, 0.0, 0.0])   # (1,0) ⊗ (1,0)

        for k in range(n):
            p = 2.0 ** (n - 1 - k)
            e2 = np.exp(-alpha * p)            # e_k^2 = exp(-alpha * p_k)

            T = np.eye(4)
            T += e2 * np.array([
                [1, p,  p,  p * p],
                [0, 1,  0,      p],
                [0, 0,  1,      p],
                [0, 0,  0,      1],
            ])
            w = w @ T

        return w[3]                            # (0,1) ⊗ (0,1) → index 3

    @staticmethod
    def _analytical_mps_2p_orbital_mps_tensors(n, alpha):
        """Construct MPS tensors for |psi> = (1/N) sum_r r*exp(-alpha*r/2) |r>.

        Bond dimension is exactly 2 for any n.
        Cost: O(n) — no exponential state vector needed.

        Same transfer-matrix structure as 1s Jacobian but with
        e_k = exp(-alpha*p_k/2) instead of exp(-alpha*p_k):

            A^{[k]0} = I_2
            A^{[k]1} = e_k * [[1, w_k], [0, 1]]   with w_k = p_k / N

        Left boundary: (1, 0),  Right boundary: (0, 1)^T
        Normalization absorbed into w_k.
        """
        norm = np.sqrt(Sto2P._norm_sq_2p_orbital(n, alpha))

        if n == 1:
            e0 = np.exp(-alpha / 2.0)
            A = np.zeros((1, 2, 1))
            A[0, 0, 0] = 0.0              # f(0) = 0 * exp(0) = 0
            A[0, 1, 0] = e0 / norm        # f(1) = 1 * exp(-alpha/2)
            return [A]

        tensors = []
        for k in range(n):
            p_k = 2.0 ** (n - 1 - k)
            e_k = np.exp(-alpha / 2.0 * p_k)
            w_k = p_k / norm

            if k == 0:
                # Left boundary: absorb row vector (1, 0)
                A = np.zeros((1, 2, 2))
                A[0, 0, :] = [1.0, 0.0]
                A[0, 1, :] = [e_k, e_k * w_k]
            elif k == n - 1:
                # Right boundary: absorb column vector (0, 1)^T
                A = np.zeros((2, 2, 1))
                A[:, 0, 0] = [0.0, 1.0]
                A[:, 1, 0] = [e_k * w_k, e_k]
            else:
                A = np.zeros((2, 2, 2))
                A[:, 0, :] = np.eye(2)
                A[:, 1, :] = [[e_k, e_k * w_k],
                              [0.0,        e_k]]

            tensors.append(A)
        return tensors

    # ------------------------------------------------------------------ #
    #  2p Jacobian  —  f(r) = r^2 * exp(-alpha*r/2)                      #
    #  Same interior transfer matrices as 2s Jacobian (chi=3),            #
    #  only the right boundary changes: (0,0,1) instead of (0,2,-alpha).  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _norm_sq_2p_jacobian(n, alpha):
        """Compute sum_{r=0}^{2^n-1} r^4 * exp(-alpha*r) in O(n)
        via 9x9 transfer matrix contraction (chi=3, so chi^2=9).

        Identical to _norm_sq_2s_jacobian except the right boundary
        is (0, 0, 1) instead of (0, 2, -alpha), extracting r^2 rather
        than r*(2-alpha*r).
        """
        l = np.array([1.0, 0.0, 0.0])
        r = np.array([0.0, 0.0, 1.0])     # extracts r^2 component

        v = np.kron(l, l)                  # 9-dim left boundary vector

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
    def _analytical_mps_2p_jacobian_mps_tensors(n, alpha):
        """Construct MPS tensors for |psi> = (1/N) sum_r r^2*exp(-alpha*r/2) |r>.

        Bond dimension is exactly 3 for any n.
        Cost: O(n) — no exponential state vector needed.

        The three bond states track the running sums:
            state 0: constant  (1)
            state 1: linear    (r_acc)
            state 2: quadratic (r_acc^2)

        Transfer matrices (identical to 2s Jacobian):
            A^{[k]0} = I_3
            A^{[k]1} = e_k * [[1, p_k, p_k^2], [0, 1, 2*p_k], [0, 0, 1]]

        Left boundary:  (1, 0, 0)          — same as 2s Jacobian
        Right boundary: (0, 0, 1)^T        — extracts r^2  (2s uses (0, 2, -alpha))
        Normalization absorbed into the left boundary (k=0 tensor).
        """
        norm = np.sqrt(Sto2P._norm_sq_2p_jacobian(n, alpha))

        if n == 1:
            e0 = np.exp(-alpha / 2.0)
            A = np.zeros((1, 2, 1))
            A[0, 0, 0] = 0.0              # f(0) = 0^2 * exp(0) = 0
            A[0, 1, 0] = e0 / norm        # f(1) = 1^2 * exp(-alpha/2)
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
                # Right boundary: absorb (0, 0, 1)^T
                # A^{[k]0} * (0,0,1)^T = I * (0,0,1)^T = (0,0,1)^T
                # A^{[k]1} * (0,0,1)^T = e_k*(p_k^2, 2*p_k, 1)^T
                A = np.zeros((3, 2, 1))
                A[:, 0, 0] = [0.0, 0.0, 1.0]
                A[:, 1, 0] = [e_k * p_k * p_k,
                              2.0 * e_k * p_k,
                              e_k]
            else:
                # Interior: identity for s=0, upper-triangular for s=1
                A = np.zeros((3, 2, 3))
                A[:, 0, :] = np.eye(3)
                A[:, 1, :] = [[e_k, e_k * p_k,      e_k * p_k * p_k],
                              [0.0,       e_k,      2.0 * e_k * p_k],
                              [0.0,       0.0,                  e_k]]

            tensors.append(A)
        return tensors
