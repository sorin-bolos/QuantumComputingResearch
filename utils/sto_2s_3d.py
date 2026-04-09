from qiskit import QuantumCircuit
import numpy as np
from scipy.linalg import null_space

from utils.arithmetic import ArithmeticOperator
from utils.matrix_product_states import Mps


class Sto2S3D:
    def __init__(self):
        self.mps = Mps()

    def sto_2s_3d_cartesian(self, qubits_per_coord, decay_constant):
        return self._sto_2s_3d_cartesian(qubits_per_coord, decay_constant)

    def sto_2s_3d_cartesian_bounded(self, qubits_per_coord, decay_constant, max_range):
        return self._sto_2s_3d_cartesian_bounded(qubits_per_coord, decay_constant, max_range)

    def split_run_results_by_coordinate(self, qubits_per_coord, data_amplitudes):
        """Extract per-coordinate marginal distributions from a 3D amplitude-encoded statevector.

        The statevector encodes psi = sum_{x,y,z} f(x,y,z)|x>|y>|z> with index ordering
        idx = jx * N^2 + jy * N + jz (matching _build_cartesian_state).

        Parameters
        ----------
        data_amplitudes : array-like
            Statevector of length 2^(3*qubits_per_coord).
        qubits_per_coord : int
            Number of qubits per spatial coordinate.

        Returns
        -------
        x_marginal, y_marginal, z_marginal : ndarray of length N = 2^qubits_per_coord
            Marginal probability distributions for each coordinate,
            computed as sum of |psi|^2 over the other two coordinates.
            Suitable for plotting as bar graphs.
        """
        N = 2 ** qubits_per_coord
        psi = np.array(data_amplitudes).reshape((N, N, N))
        probs = np.abs(psi) ** 2
        x_marginal = probs.sum(axis=(1, 2))
        y_marginal = probs.sum(axis=(0, 2))
        z_marginal = probs.sum(axis=(0, 1))
        return x_marginal, y_marginal, z_marginal

    def _sto_2s_3d_cartesian(self, qubits_per_coord, decay_constant):
        n_total = 3 * qubits_per_coord

        psi = self._build_cartesian_state(qubits_per_coord, decay_constant)
        return self.mps.generate_mps_circuit_from_amplitudes(psi, n_total)
    
    def _sto_2s_3d_cartesian_bounded(self, qubits_per_coord, decay_constant, max_range):
        n_total = 3 * qubits_per_coord

        psi = self._build_cartesian_state_bounded(qubits_per_coord, decay_constant, max_range)
        return self.mps.generate_mps_circuit_from_amplitudes(psi, n_total)

    def _build_cartesian_state(self, qubits_per_coord, alpha, threshold=1e-15):
        """Build the 3D state vector for a hydrogen orbital on a Cartesian grid.

        Parameters
        ----------
        qubits_per_coord : int   — qubits per coordinate (total qubits = 3*qubits_per_coord)
        alpha       : float — Bohr radius parameter

        Returns
        -------
        psi : ndarray of length 2^(3n), normalized
        """
        N = 2 ** qubits_per_coord
        half = N // 2  # offset so coordinates are centered: coord = j - half

        psi = np.zeros(N ** 3)
        idx = 0
        for jx in range(N):
            x = jx - half
            for jy in range(N):
                y = jy - half
                for jz in range(N):
                    z = jz - half
                    r = np.sqrt(x*x + y*y + z*z)
                    psi[idx] = (2.0 - r * alpha) * np.exp(-alpha*r/2.0)
                    idx += 1

        norm = np.linalg.norm(psi)
        if norm > threshold:
            psi /= norm
        return psi
    
    def _build_cartesian_state_bounded(self, qubits_per_coord, alpha, max_range, threshold=1e-15):
        """Build the 3D state vector for a hydrogen orbital on a Cartesian grid.

        Parameters
        ----------
        qubits_per_coord : int   — qubits per coordinate (total qubits = 3*qubits_per_coord)
        alpha       : float — Bohr radius parameter (in the same physical units as max_range)
        max_range   : float — Total physical size of the space per coordinate.
                              Grid spans [-max_range/2, +max_range/2) on each axis.
                              Increasing qubits_per_coord increases resolution, not range.

        Returns
        -------
        psi : ndarray of length 2^(3n), normalized
        """
        N = 2 ** qubits_per_coord
        half = N // 2
        step = max_range / N  # physical spacing between grid points

        psi = np.zeros(N ** 3)
        idx = 0
        for jx in range(N):
            x = (jx - half) * step
            for jy in range(N):
                y = (jy - half) * step
                for jz in range(N):
                    z = (jz - half) * step
                    r = np.sqrt(x*x + y*y + z*z)
                    psi[idx] = (2.0 - r * alpha) * np.exp(-alpha*r/2.0)
                    idx += 1

        norm = np.linalg.norm(psi)
        if norm > threshold:
            psi /= norm
        return psi