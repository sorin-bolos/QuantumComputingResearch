from utils.circuit_handler import CircuitHandler

class Integals:

    def __init__(self, allow_measurements=True, optimize_t_gates=True):
        self.allow_measurements = allow_measurements
        self.optimize_t_gates = optimize_t_gates
        self.ch = CircuitHandler()

    def get_s1_1d_overlap_circuit(self, qubit_count, decay_constant, max_range, center_distance):
        from utils.sto_1s_1d import Sto1S
        
        s1_generator = Sto1S(self.allow_measurements)
        s1_1d_1 = s1_generator.get_sto_1s_1d_carthesian(qubit_count, decay_constant, max_range, center_distance)
        s1_1d_2 = s1_generator.get_sto_1s_1d_carthesian_dagger(qubit_count, decay_constant, max_range, 0)

        return self.ch.concatenate(s1_1d_1, s1_1d_2)

