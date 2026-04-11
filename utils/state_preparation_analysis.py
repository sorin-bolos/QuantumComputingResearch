import numpy as np

from utils.matrix_product_states import Mps
from utils.sto_1s_3d import Sto1S3D
from utils.sto_2s_3d import Sto2S3D

class StatePreparationAnalysis:
    def __init__(self):
        self.mps = Mps()

    def analyze_1d(self, max_qubit_count, max_range):
        decay_constant = 1.0
        center_offset = max_range / 2

        func_1s_1d_uncentered = lambda x: np.exp(-decay_constant*np.abs(x))  # 1s orbital shape (unnormalized)
        self._chi_analysis_1d("1s 1D Orbital Uncentered", func_1s_1d_uncentered, max_qubit_count, max_range)

        func_2s_1d_sto_uncentered = lambda x: np.abs(x)*np.exp(-decay_constant*np.abs(x))  # 2s orbital shape (unnormalized)
        self._chi_analysis_1d("2s 1D STO Uncentered", func_2s_1d_sto_uncentered, max_qubit_count, max_range)

        func_2s_1d_h2_uncentered = lambda x: (2 - decay_constant*np.abs(x))*np.exp(-decay_constant*np.abs(x)/2)  # 2s orbital shape (unnormalized)
        self._chi_analysis_1d("2s 1D H2 Uncentered", func_2s_1d_h2_uncentered, max_qubit_count, max_range)


        func_1s_1d = lambda x: np.exp(-decay_constant*np.abs(x - center_offset))  # 1s orbital shape (unnormalized)
        self._chi_analysis_1d("1s 1D Orbital", func_1s_1d, max_qubit_count, max_range)

        func_2s_1d_sto = lambda x: np.abs(x-center_offset)*np.exp(-decay_constant*np.abs(x - center_offset))  # 2s orbital shape (unnormalized)
        self._chi_analysis_1d("2s 1D STO", func_2s_1d_sto, max_qubit_count, max_range)

        func_2s_1d_h2 = lambda x: (2 - decay_constant*np.abs(x-center_offset))*np.exp(-decay_constant*np.abs(x-center_offset)/2)  # 2s orbital shape (unnormalized)
        self._chi_analysis_1d("2s 1D H2", func_2s_1d_h2, max_qubit_count, max_range)

        func_2s_1d_h2_jacobian = lambda x: x*(2 - decay_constant*x)*np.exp(-decay_constant*x/2)  # 2s orbital shape with jacobian (unnormalized)
        self._chi_analysis_1d("2s 1D H2 with Jacobian", func_2s_1d_h2_jacobian, max_qubit_count, max_range)
        
        potential_offset = center_offset + 1.4
        def potential_times_s1(x):
            s1 = np.exp(-decay_constant * np.abs(x - center_offset))
            V = 1 / np.abs(x - potential_offset)
            f = V * s1
            return f
        self._chi_analysis_1d("1s 1D Orbital times Coulomb potential", potential_times_s1, max_qubit_count, max_range)

    def _chi_analysis_1d(self, function_name, function_1d, max_qubit_count, max_range):
        print()
        print(f"############# {function_name}        Range={max_range} #############")
        print(f"{'Space range [a.u.]':>15s}  {'qubit count':>15s}  {'max χ':>6s}")
        print("─" * 50)

        for n in range(2, max_qubit_count + 1):
            max_chi = self.mps.compute_max_bond_dimension(function_1d, n, max_range)
            print(f"{max_range:>15.1f}  {n:>15d}    {max_chi:>6d}")
        print()

    def bond_dimension_sensitivity_to_decay_constant(self):

        for o in [Sto1S3D(), Sto2S3D()]:
            for s in [32, 64]:
                for n in range(4, 8):
                    self._bond_dimension_sensitivity_to_decay_constant(o, n, s)

    def analyze_3d_orbitals(self):

        decay_constant = 1.0
        for o in [Sto1S3D(), Sto2S3D()]:
            for s in [32, 64]:
                self._analyze_3d_orbitals(o, decay_constant, s)

    def _bond_dimension_sensitivity_to_decay_constant(self, sto, qubit_count_per_coord, max_range):
        orbital = "1S" if isinstance(sto, Sto1S3D) else "2S"

        print(f"Bond dimension sensitivity of {orbital} to a  (n_per_coord = {qubit_count_per_coord}, {3 * qubit_count_per_coord} qubits, {2 ** qubit_count_per_coord}³ grid)\n")
        print(f"{'Orbital':>10s}    {'Space range [a.u.]':>15s}  {'qubits per coordonate':>25s}     {'alpha':>8s}    {'max χ':>6s}")
        print("─" * 100)
        
        for a_val in np.linspace(0.1, 6.0, 30):
            max_bond = sto.get_max_bond_dimension(qubit_count_per_coord, a_val, max_range)
            print(f"{orbital:>10s}    {max_range:>15.1f}  {qubit_count_per_coord:>20d}             {a_val:>8.2f}    {max_bond:>6d} ")
        print()
    
    def _analyze_3d_orbitals(self, sto, decay_constant, max_range):
        import time

        print("Scaling of bond dimensions with grid resolution\n")
        print(f"{'Orbital':>10s}    {'Space range [a.u.]':>15s}  {'qubits per coordonate':>25s}     {'alpha':>8s}    {'max χ':>6s}  {'time':>8s}")
        print("─" * 100)

        orbital = "1S" if isinstance(sto, Sto1S3D) else "2S"

        for n_pc in range(4, 11):
            t0 = time.perf_counter()
            max_bond = sto.get_max_bond_dimension(n_pc, decay_constant, max_range)
            dt = time.perf_counter() - t0
            print(f"{orbital:>10s}    {max_range:>15.1f}  {n_pc:>20d}             {decay_constant:>8.2f}    {max_bond:>6d}  {dt:>7.2f}s")
        
        print()