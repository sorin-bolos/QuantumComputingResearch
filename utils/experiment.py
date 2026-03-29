from utils.integrals import Integals
from utils.simulation_excutor import SimulationExecutor
from utils.sample_interpreter import SampleInterpreter
from utils.noisy_simulation_executor import NoisySimulationExecutor
from utils.noisy_estimator_executor import NoisyEstimatorExecutor
from utils.ibm_executor import IbmExecutor
from utils.ibm_estimator_executor import IbmEstimatorExecutor
from utils.dataclasses import Results


class Experiment:

    def __init__(
        self,
        allow_measurement: bool = True,
        optimize_t_gates: bool = True,
        enable_dd: bool = True,
        enable_twirling: bool = True,
        enable_measure_mitigation: bool = True,
        enable_zne: bool = True,
        ibm_backend=None,
    ):
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates
        self.ibm_backend = ibm_backend

        # DD is incompatible with dynamic circuits (mid-circuit measurements).
        # When allow_measurement=True the circuit uses if_else, so DD is silently
        # disabled regardless of the enable_dd flag.
        enable_dd = enable_dd and (not allow_measurement)

        self._noisy_simulation_executor = NoisySimulationExecutor(
            enable_dd=enable_dd,
            enable_twirling=enable_twirling,
            enable_m3=True,
        )
        self._noisy_estimator_executor = NoisyEstimatorExecutor(
            enable_dd=enable_dd,
            enable_twirling=enable_twirling,
            enable_measure_mitigation=enable_measure_mitigation,
            enable_zne=enable_zne,
            zne_noise_factors=[1, 3, 5, 7],
        )

        if ibm_backend is not None:
            self.ibm_executor = IbmExecutor(
                ibm_backend,
                enable_dd=enable_dd,
                enable_twirling=enable_twirling,
                enable_m3=True,
            )
            self.ibm_estimator_executor = IbmEstimatorExecutor(
                ibm_backend,
                enable_dd=enable_dd,
                enable_twirling=enable_twirling,
                enable_measure_mitigation=enable_measure_mitigation,
                enable_zne=enable_zne,
                zne_noise_factors=[1, 3, 5, 7],
            )
        else:
            self.ibm_executor = None
            self.ibm_estimator_executor = None

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

        integrals = Integals(self.allow_measurement, self.optimize_t_gates)
        qc = integrals.get_s1_1d_overlap_circuit(qubit_count, decay_constant, max_range, center_distance)

        # ── noiseless simulation ───────────────────────────────────────────────

        if print_results:
            print("############# Noiseless simulation #############")
            print()

        data_amps = simulation_executor.get_data_amplitudes(qc, qubit_count)
        counts = simulation_executor.sample_measurement_counts(qc, qubit_count, shots=shots)

        analitical_zero_amplitude = simulation_executor.get_analytical_zero_amplitude(data_amps)
        analitical_zero_probablity = simulation_executor.get_analytical_zero_probability(data_amps)
        sampled_zero_amplitude = sample_interpreter.get_zero_amplitude(counts)
        sampled_zero_probability = sample_interpreter.get_zero_probability(counts)

        if print_results:
            print(f"analitical_zero_amplitude  = {analitical_zero_amplitude}")
            print(f"analitical_zero_probability = {analitical_zero_probablity}")
            print(f"sampled_zero_amplitude     = {sampled_zero_amplitude}")
            print(f"sampled_zero_probability   = {sampled_zero_probability}")
            sample_interpreter.print_errors(0.600494, sampled_zero_amplitude, analitical_zero_amplitude)

        # ── noisy simulation (sampler) ─────────────────────────────────────────

        if print_results:
            print()
            print("############# Noisy simulation #############")
            print()

        self._noisy_simulation_executor.print_circuit_stats(qc)
        counts = self._noisy_simulation_executor.sample_measurement_counts(qc, qubit_count, shots=shots)
        noisy_sampled_zero_amplitude = sample_interpreter.get_zero_amplitude(counts)
        noisy_sampled_zero_probability = sample_interpreter.get_zero_probability(counts)

        if print_results:
            print(f"noisy_sampled_zero_amplitude  = {noisy_sampled_zero_amplitude}")
            print(f"noisy_sampled_zero_probability = {noisy_sampled_zero_probability}")
            sample_interpreter.print_errors(0.600494, noisy_sampled_zero_amplitude, analitical_zero_amplitude)

        # ── noisy simulation (estimator + ZNE) ────────────────────────────────

        if print_results:
            print()
            print("############# Noisy estimator #############")
            print()

        estimator_zero_probability = self._noisy_estimator_executor.get_probability_of_zero(qc, qubit_count, shots=shots)
        estimator_zero_amplitude = self._noisy_estimator_executor.get_amplitude_of_zero(qc, qubit_count, shots=shots)

        if print_results:
            print(f"estimator_zero_probability = {estimator_zero_probability}")
            print(f"estimator_zero_amplitude   = {estimator_zero_amplitude}")
            sample_interpreter.print_errors(0.600494, estimator_zero_amplitude, analitical_zero_amplitude)

        # ── real hardware ──────────────────────────────────────────────────────

        ibm_sampler_zero_amplitude = None
        ibm_sampler_zero_probability = None
        ibm_estimator_zero_amplitude = None
        ibm_estimator_zero_probability = None

        if self.ibm_backend is not None:
            if print_results:
                print()
                print("############# IBM hardware (sampler) #############")
                print()

            self.ibm_executor.print_circuit_stats(qc)
            ibm_counts = self.ibm_executor.sample_measurement_counts(qc, qubit_count, shots=shots)
            ibm_sampler_zero_amplitude = sample_interpreter.get_zero_amplitude(ibm_counts)
            ibm_sampler_zero_probability = sample_interpreter.get_zero_probability(ibm_counts)

            if print_results:
                print(f"ibm_sampler_zero_amplitude  = {ibm_sampler_zero_amplitude}")
                print(f"ibm_sampler_zero_probability = {ibm_sampler_zero_probability}")
                sample_interpreter.print_errors(0.600494, ibm_sampler_zero_amplitude, analitical_zero_amplitude)

            if print_results:
                print()
                print("############# IBM hardware (estimator + ZNE) #############")
                print()

            ibm_estimator_zero_probability = self.ibm_estimator_executor.get_probability_of_zero(qc, qubit_count, shots=shots)
            ibm_estimator_zero_amplitude = self.ibm_estimator_executor.get_amplitude_of_zero(qc, qubit_count, shots=shots)

            if print_results:
                print(f"ibm_estimator_zero_probability = {ibm_estimator_zero_probability}")
                print(f"ibm_estimator_zero_amplitude   = {ibm_estimator_zero_amplitude}")
                sample_interpreter.print_errors(0.600494, ibm_estimator_zero_amplitude, analitical_zero_amplitude)

        return Results(
            analitical_zero_amplitude=analitical_zero_amplitude,
            analitical_zero_probablity=analitical_zero_probablity,
            sampled_zero_amplitude=sampled_zero_amplitude,
            sampled_zero_probability=sampled_zero_probability,
            noisy_sampled_zero_amplitude=noisy_sampled_zero_amplitude,
            noisy_sampled_zero_probability=noisy_sampled_zero_probability,
            estimator_zero_amplitude=estimator_zero_amplitude,
            estimator_zero_probability=estimator_zero_probability,
            ibm_sampler_zero_amplitude=ibm_sampler_zero_amplitude,
            ibm_sampler_zero_probability=ibm_sampler_zero_probability,
            ibm_estimator_zero_amplitude=ibm_estimator_zero_amplitude,
            ibm_estimator_zero_probability=ibm_estimator_zero_probability,
        )
