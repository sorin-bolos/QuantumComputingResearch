import numpy as np

from utils.matrix_product_states import Mps

class StatePreparationAnalysis:
    def __init__(self):
        self.mps = Mps()

    def analyze(self, decay_constant, max_qubit_count, max_range):
        func_1s_1d = lambda x, a: np.exp(-a*x)  # 1s orbital shape (unnormalized)
        self.chi_analysis("1s 1D Orbital", func_1s_1d, decay_constant, max_qubit_count, max_range)

        func_2s_1d_sto = lambda x, a: x*np.exp(-a*x)  # 2s orbital shape (unnormalized)
        self.chi_analysis("2s 1D STO", func_2s_1d_sto, decay_constant, max_qubit_count, max_range)

        func_2s_1d_h2 = lambda x, a: (2 - a*x)*np.exp(-a*x/2)  # 2s orbital shape (unnormalized)
        self.chi_analysis("2s 1D H2", func_2s_1d_h2, decay_constant, max_qubit_count, max_range)

        func_2s_1d_h2_jacobian = lambda x, a: x*(2 - a*x)*np.exp(-a*x/2)  # 2s orbital shape with jacobian (unnormalized)
        self.chi_analysis("2s 1D H2 with Jacobian", func_2s_1d_h2_jacobian, decay_constant, max_qubit_count, max_range)

        # func_1s_3d = 

    def chi_analysis_1d(self, function_name, function_1d, decay_constant, max_qubit_count, max_range):
        print()
        print(f"############# {function_name} #############")

        for n in range(2, max_qubit_count + 1):
            scale = (2 ** n) / max_range
            scaled_decay_constant = decay_constant / scale

            func = lambda x: function_1d(x, scaled_decay_constant)

            max_chi = self.mps.compute_max_bond_dimension(func, n, max_range)
            print(f"n={n} qubits: max bond dimension = {max_chi}")
        print()

    def analyze_3d_orbitals(self, decay_constant, max_qubit_count, max_range):
        import time

        print("Scaling of bond dimensions with grid resolution\n")
        print(f"{'Orbital':>8s}  {'n/c':>4s}  {'3n':>4s}  "
            f"{'max χ':>6s}  {'time':>8s} { 'alpha':>7s}")
        print("─" * 90)

        for n_pc in range(2, max_qubit_count + 1):
            
            scale = (2 ** n_pc) / max_range
            scaled_decay_constant = decay_constant / scale

            n_total = 3 * n_pc
            for orbital in ['1S', '2S']:
                t0 = time.perf_counter()
                psi = self._build_cartesian_state(n_pc, scaled_decay_constant, orbital)
                max_bond = self.mps.compute_max_bond_dimension_from_amplitudes(psi, n_total)
                dt = time.perf_counter() - t0
                print(f"{orbital:>8s}  {n_pc:>4d}  {n_total:>4d}  "
                    f"{max_bond:>6d}  {dt:>7.2f}s {scaled_decay_constant:>7.2f}")
                
    def bond_dimension_sensitivity_to_decay_constant(self, qubit_count_per_coord):
        print(f"Bond dimension sensitivity to a  (n_per_coord = {qubit_count_per_coord}, {3 * qubit_count_per_coord} qubits, {2 ** qubit_count_per_coord}³ grid)\n")
        print(f"{'Orbital':>8s}  {'a':>6s}  {'max χ':>6s}  {'χ(x|y)':>7s}  {'full bond profile'}")
        print("─" * 75)

        n_pc = qubit_count_per_coord
        n_total = 3 * n_pc
        for orbital in ['1S', '2S']:
            for a_val in np.linspace(0.1, 6.0, 30):
                psi = self._build_cartesian_state(n_pc, a_val, orbital)
                tensors = self.mps._decompose_to_mps_right_canonical(n_total, psi)
                bonds = [tensors[k].shape[2] for k in range(n_total - 1)]
                print(f"{orbital:>8s}  {a_val:>6.1f}  {max(bonds):>6d}  {bonds[n_pc-1]:>7d}  {bonds}")
            print()

    def _build_cartesian_state(self, n_per_coord, a, orbital='1S'):
        """Build the 3D state vector for a hydrogen orbital on a Cartesian grid.

        Parameters
        ----------
        n_per_coord : int   — qubits per coordinate (total qubits = 3*n_per_coord)
        a           : float — Bohr radius parameter
        orbital     : '1S' or '2S'

        Returns
        -------
        psi : ndarray of length 2^(3n), normalized
        """
        N = 2 ** n_per_coord
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
                    if orbital == '1S':
                        psi[idx] = np.exp(-r * a)
                    elif orbital == '2S':
                        psi[idx] = (2.0 - r * a) * np.exp(-r * a / (2.0))
                    idx += 1

        norm = np.linalg.norm(psi)
        if norm > 1e-15:
            psi /= norm
        return psi