from utils.integrals import Integals
from utils.simulation_excutor import SimulationExecutor
from utils.sample_interpreter import SampleInterpreter
from utils.noisy_simulation_executor import NoisySimulationExecutor
from utils.dataclasses import Results

class Experiment:

    def __init__(self, allow_measurement: bool = True, optimize_t_gates: bool =True):
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates

    def run_single_s1_1d_overlap_integral(
            self, 
            qubit_count: int, 
            decay_constant: float, 
            max_range: int, 
            center_distance: int,
            shots: int = 1024,
            print_results: bool = True) -> Results:
        
        simulation_executor = SimulationExecutor()
        sample_interpreter = SampleInterpreter()
        noisy_simlutation_executor = NoisySimulationExecutor(enable_dd=(not self.allow_measurement), enable_twirling=True, enable_m3=True)
        
        integrals = Integals(self.allow_measurement, self.optimize_t_gates)
        qc = integrals.get_s1_1d_overlap_circuit(qubit_count, decay_constant, max_range, center_distance)

        if print_results:
            print("############# Noisless simulation ############3")
            print()

        data_amps = simulation_executor.get_data_amplitudes(qc, qubit_count)
        counts = simulation_executor.sample_measurement_counts(qc, qubit_count, shots=shots)

        analitical_zero_amplitude = simulation_executor.get_analytical_zero_amplitude(data_amps)
        analitical_zero_probablity = simulation_executor.get_analytical_zero_probability(data_amps)
        sampled_zero_amplitude = sample_interpreter.get_zero_amplitude(counts)
        sampled_zero_probability = sample_interpreter.get_zero_probability(counts)

        if print_results:
            print(f"analitical_zero_amplitude = {analitical_zero_amplitude}")
            print(f"analitical_zero_probability = {analitical_zero_probablity}")
            print(f"sampled_zero_amplitude = {sampled_zero_amplitude}")
            print(f"sampled_zero_probability = {sampled_zero_probability}")

            sample_interpreter.print_errors(0.600494, sampled_zero_amplitude, analitical_zero_amplitude)

            print("############# Noisy simulation ############3")
            print()

        noisy_simlutation_executor.print_circuit_stats(qc)
        counts = noisy_simlutation_executor.sample_measurement_counts(qc, qubit_count, shots=shots)
        noisy_sampled_zero_amplitude = sample_interpreter.get_zero_amplitude(counts)
        noisy_sampled_zero_probability = sample_interpreter.get_zero_probability(counts)

        if print_results:
            print(f"noisy_sampled_zero_amplitude = {noisy_sampled_zero_amplitude}")
            print(f"noisy_sampled_zero_probability = {noisy_sampled_zero_probability}")

            sample_interpreter.print_errors(0.600494, noisy_sampled_zero_amplitude, analitical_zero_amplitude)

        return Results(
            analitical_zero_amplitude=analitical_zero_amplitude,
            analitical_zero_probablity=analitical_zero_probablity,
            sampled_zero_amplitude=sampled_zero_amplitude,
            sampled_zero_probability=sampled_zero_probability,
            noisy_sampled_zero_amplitude=noisy_sampled_zero_amplitude,
            noisy_sampled_zero_probability=noisy_sampled_zero_probability
        )

