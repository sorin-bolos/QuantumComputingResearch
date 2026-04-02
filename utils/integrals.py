from utils.circuit_handler import CircuitHandler
import numpy as np

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
