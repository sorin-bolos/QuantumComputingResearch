from utils.circuit_handler import CircuitHandler
import numpy as np
from scipy.integrate import quad
from scipy.special import k0

class Integals:

    def __init__(self, allow_measurements=True, optimize_t_gates=True):
        self.allow_measurements = allow_measurements
        self.optimize_t_gates = optimize_t_gates
        self.ch = CircuitHandler()

    def get_1s_1d_overlap_exact_result(self, decay_constant, center_distance):
        exact_result = (1 + center_distance) * np.exp(-center_distance)
        return exact_result
    
    def get_1s_1d_kinetic_exact_result(self, decay_constant, center_distance):
        exact_result = 0.5 * decay_constant ** 2 * (1 - decay_constant * center_distance) * np.exp(-decay_constant * center_distance)
        return exact_result
    
    def get_V11A_1D_exact(self, decay_constant, eps=0.01):
        # own nucleus: <phi_A | -v(x) | phi_A>  with phi_A centred at 0, nucleus at 0
        # Analytical: -2ζ · K₀(2ζ√ε)  from ∫₀^∞ e^{-at}/√(t²+b²) dt = K₀(ab), a=2ζ, b=√ε
        return -2 * decay_constant * k0(2 * decay_constant * np.sqrt(eps))

    def get_V11B_1D_exact(self, decay_constant, center_distance, eps=0.01):
        # other nucleus: <phi_A | -v(x-d) | phi_A>  with phi_A centred at 0, nucleus at d
        psi1d    = lambda x: decay_constant * np.exp(-2 * decay_constant * np.abs(x))
        softcore = lambda x: 1.0 / np.sqrt((x - center_distance)**2 + eps)
        f = lambda x: -psi1d(x) * softcore(x)
        return quad(f, -60, 60, points=[0.0, center_distance], limit=400)[0]

    def get_V12_1D_exact(self, decay_constant, center_distance, eps=0.01):
        # cross term: <phi_A | -v(x) | phi_B> + <phi_A | -v(x-d) | phi_B>
        # Both nuclei contribute; V12^A = V12^B by homonuclear symmetry so total = 2 * V12^A
        psi_A    = lambda x: np.sqrt(decay_constant) * np.exp(-decay_constant * np.abs(x))
        psi_B    = lambda x: np.sqrt(decay_constant) * np.exp(-decay_constant * np.abs(x - center_distance))
        softcoreA = lambda x: 1.0 / np.sqrt(x**2 + eps)
        f = lambda x: -psi_A(x) * softcoreA(x) * psi_B(x)
        V12A, _ = quad(f, -60, 60, points=[0.0, center_distance], limit=400)
        return 2 * V12A

    def get_s1_1d_overlap_circuit(self, qubit_count, decay_constant, center_distance):
        from utils.sto_1s_1d import Sto1S

        s1_generator = Sto1S(self.allow_measurements, self.optimize_t_gates)
        s1_1d_1 = s1_generator.get_sto_1s_1d_carthesian(qubit_count, decay_constant, center_distance)
        s1_1d_2 = s1_generator.get_sto_1s_1d_carthesian_dagger(qubit_count, decay_constant, 0)

        return self.ch.concatenate(s1_1d_1, s1_1d_2)

    # Kinetic energy integrals  T₁₂ = <φ₁| -½∇² |φ₂>      
    def get_s1_1d_kinetic_derivative(self, qubit_count, decay_constant, center_distance):
        """Variant 3 – Integration by parts via derivative states.

        Uses  T₁₂ = ½ <∂φ₁|∂φ₂>  where  ∂φ(x) = d/dx[e^(-a|x-r|)].

        The normalised derivative state |∂φ̃> satisfies ||∂φ|| = a·||φ||, so the
        prefactor is simply ½a² regardless of individual normalisations:

            T₁₂ = ½ a² · amplitude(circuit)

        Returns
        -------
        circuit : QuantumCircuit
            Circuit whose |0⟩ amplitude gives <∂φ̃₁|∂φ̃₂>.
        scale : float
            ½ a²
        """
        from utils.sto_1s_1d import Sto1S

        s1_generator = Sto1S(self.allow_measurements, self.optimize_t_gates)
        s1_1d_1 = s1_generator.get_sto_1s_1d_derivative_carthesian(qubit_count, decay_constant, center_distance)
        s1_1d_2 = s1_generator.get_sto_1s_1d_derivative_carthesian_dagger(qubit_count, decay_constant, 0)

        qc =  self.ch.concatenate(s1_1d_1, s1_1d_2)
        print("Constructed kinetic derivative circuit:")
        print(type(qc))
        return qc
    
    def get_V11A_1D_circuit(self, qubit_count, scaled_decay_constant, decay_constant, max_range):
        from utils.sto_1s_1d import Sto1S

        scaled_center_offset = 2**qubit_count // 2

        s1_generator = Sto1S(self.allow_measurements, self.optimize_t_gates)
        V11A_circuit = s1_generator.get_sto_1s_1d_potential_cartesian(8, 1, 8, 8, 16) #(qubit_count, decay_constant, max_range/2, max_range/2, max_range)
        # V11A_circuit_dagger = s1_generator.get_sto_1s_1d_potential_cartesian_dagger(qubit_count, decay_constant, max_range/2, max_range/2, max_range)
        # s1_1d = s1_generator.get_sto_1s_1d_carthesian(qubit_count, scaled_decay_constant, scaled_center_offset)
        s1_1d_d = s1_generator.get_sto_1s_1d_carthesian_dagger(8, 1/16, 128)#(qubit_count, scaled_decay_constant, scaled_center_offset)

        return self.ch.concatenate(V11A_circuit, s1_1d_d)

    def get_V11B_1D_circuit(self, qubit_count, scaled_decay_constant, decay_constant, max_range):
        from utils.sto_1s_1d import Sto1S

        scaled_center_offset = 2**qubit_count // 2

        s1_generator = Sto1S(self.allow_measurements, self.optimize_t_gates)
        V11A_circuit = s1_generator.get_sto_1s_1d_potential_cartesian(8, 1, 8, 9.375, 16) #(qubit_count, decay_constant, max_range/2, max_range/2, max_range)
        # V11A_circuit_dagger = s1_generator.get_sto_1s_1d_potential_cartesian_dagger(qubit_count, decay_constant, max_range/2, max_range/2, max_range)
        # s1_1d = s1_generator.get_sto_1s_1d_carthesian(qubit_count, scaled_decay_constant, scaled_center_offset)
        s1_1d_d = s1_generator.get_sto_1s_1d_carthesian_dagger(8, 1/16, 128)#(qubit_count, scaled_decay_constant, scaled_center_offset)

        return self.ch.concatenate(V11A_circuit, s1_1d_d)

    def get_V12_1_1D_circuit(self, qubit_count, scaled_decay_constant, decay_constant, max_range):
        from utils.sto_1s_1d import Sto1S

        scaled_center_offset = 2**qubit_count // 2

        s1_generator = Sto1S(self.allow_measurements, self.optimize_t_gates)
        V11A_circuit = s1_generator.get_sto_1s_1d_potential_cartesian(8, 1, 9.375, 8, 16) #(qubit_count, decay_constant, max_range/2, max_range/2, max_range)
        # V11A_circuit_dagger = s1_generator.get_sto_1s_1d_potential_cartesian_dagger(qubit_count, decay_constant, max_range/2, max_range/2, max_range)
        # s1_1d = s1_generator.get_sto_1s_1d_carthesian(qubit_count, scaled_decay_constant, scaled_center_offset)
        s1_1d_d = s1_generator.get_sto_1s_1d_carthesian_dagger(8, 1/16, 128)#(qubit_count, scaled_decay_constant, scaled_center_offset)

        return self.ch.concatenate(V11A_circuit, s1_1d_d)
    
    def get_V12_2_1D_circuit(self, qubit_count, scaled_decay_constant, decay_constant, max_range):
        from utils.sto_1s_1d import Sto1S

        scaled_center_offset = 2**qubit_count // 2

        s1_generator = Sto1S(self.allow_measurements, self.optimize_t_gates)
        V11A_circuit = s1_generator.get_sto_1s_1d_potential_cartesian(8, 1, 9.375, 9.375, 16) #(qubit_count, decay_constant, max_range/2, max_range/2, max_range)
        # V11A_circuit_dagger = s1_generator.get_sto_1s_1d_potential_cartesian_dagger(qubit_count, decay_constant, max_range/2, max_range/2, max_range)
        # s1_1d = s1_generator.get_sto_1s_1d_carthesian(qubit_count, scaled_decay_constant, scaled_center_offset)
        s1_1d_d = s1_generator.get_sto_1s_1d_carthesian_dagger(8, 1/16, 128)#(qubit_count, scaled_decay_constant, scaled_center_offset)

        return self.ch.concatenate(V11A_circuit, s1_1d_d)
