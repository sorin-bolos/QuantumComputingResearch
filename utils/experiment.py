from utils.integrals import Integals
from utils.simulation_excutor import SimulationExecutor
from utils.sample_interpreter import SampleInterpreter
from utils.noisy_sampler_executor import NoisySamplerExecutor
from utils.noisy_estimator_executor import NoisyEstimatorExecutor
from utils.ibm_sampler_executor import IbmSamplerExecutor
from utils.ibm_estimator_executor import IbmEstimatorExecutor
from utils.dataclasses import Results
from utils.resource_estimator import ResourceEstimator
import numpy as np


class Experiment:

    def __init__(
        self,
        allow_measurement: bool = True,
        optimize_t_gates: bool = True,
        enable_dd: bool = True,
        enable_twirling: bool = True,
        enable_measure_mitigation: bool = True,
        enable_zne: bool = True,
        zne_noise_factors: list = None,
        zne_extrapolator: str = 'linear',
        ibm_backend=None,
        fake_backend=None,
    ):
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates
        self.ibm_backend = ibm_backend
        self.fake_backend = fake_backend

        # DD is incompatible with dynamic circuits (mid-circuit measurements).
        # When allow_measurement=True the circuit uses if_else, so DD is silently
        # disabled regardless of the enable_dd flag.
        enable_dd = enable_dd and (not allow_measurement)
        zne_noise_factors = zne_noise_factors or [1, 2, 3]

        self._noisy_sampler_executor = NoisySamplerExecutor(
            enable_dd=enable_dd,
            enable_twirling=enable_twirling,
            enable_m3=True,
            backend = fake_backend
        )
        self._noisy_estimator_executor = NoisyEstimatorExecutor(
            enable_dd=enable_dd,
            enable_twirling=enable_twirling,
            enable_measure_mitigation=enable_measure_mitigation,
            enable_zne=enable_zne,
            zne_noise_factors=zne_noise_factors,
            zne_extrapolator=zne_extrapolator,
            backend=fake_backend
        )

        if ibm_backend is not None:
            self.ibm_sampler_executor = IbmSamplerExecutor(
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
                zne_noise_factors=zne_noise_factors,
                zne_extrapolator=zne_extrapolator,
            )
        else:
            self.ibm_sampler_executor = None
            self.ibm_estimator_executor = None

        self.resource_estimator = ResourceEstimator()

    def run_single_s1_1d_overlap_integral(
            self,
            qubit_count: int,
            decay_constant: float,
            max_range: int, # the size of the space that will be represented by 2^qubit_count values
            center_distance: float, # the distance between integral centers
            shots: int = 1024) -> Results:

        scale = (2 ** qubit_count) / max_range
        scaled_center_distance = round(center_distance * scale)
        scaled_decay_constant = decay_constant / scale

        used_center_distance = scaled_center_distance / scale
        exact_result = (1 + used_center_distance) * np.exp(-used_center_distance)

        simulation_executor = SimulationExecutor()
        sample_interpreter = SampleInterpreter()

        integrals = Integals(self.allow_measurement, self.optimize_t_gates)
        qc = integrals.get_s1_1d_overlap_circuit(qubit_count, scaled_decay_constant, scaled_center_distance)

        noisy_simulation_stats = self.resource_estimator.get_circuit_stats(qc, self.fake_backend)
        ibm_backend_stats = self.resource_estimator.get_circuit_stats(qc, self.ibm_backend)

        # ── noiseless simulation ───────────────────────────────────────────────

        data_amps = simulation_executor.get_data_amplitudes(qc, qubit_count)
        counts = simulation_executor.sample_measurement_counts(qc, qubit_count, shots=shots)

        analitical_zero_amplitude = simulation_executor.get_analytical_zero_amplitude(data_amps)
        analitical_zero_probablity = simulation_executor.get_analytical_zero_probability(data_amps)
        sampled_zero_amplitude = sample_interpreter.get_zero_amplitude(counts)
        sampled_zero_probability = sample_interpreter.get_zero_probability(counts)

        # ── noisy simulation (sampler) ─────────────────────────────────────────

        counts = self._noisy_sampler_executor.sample_measurement_counts(qc, qubit_count, shots=shots)
        noisy_sampled_zero_amplitude = sample_interpreter.get_zero_amplitude(counts)
        noisy_sampled_zero_probability = sample_interpreter.get_zero_probability(counts)

        # ── noisy simulation (estimator + ZNE) ────────────────────────────────

        estimator_zero_amplitude, estimator_zero_probability = self._noisy_estimator_executor.get_amplitude_of_zero(qc, qubit_count, shots=shots)

        # ── real hardware ──────────────────────────────────────────────────────

        ibm_sampler_zero_amplitude = None
        ibm_sampler_zero_probability = None
        ibm_estimator_zero_amplitude = None
        ibm_estimator_zero_probability = None

        # if self.ibm_backend is not None:
        #     ibm_counts = self.ibm_sampler_executor.sample_measurement_counts(qc, qubit_count, shots=shots)
        #     ibm_sampler_zero_amplitude = sample_interpreter.get_zero_amplitude(ibm_counts)
        #     ibm_sampler_zero_probability = sample_interpreter.get_zero_probability(ibm_counts)

        #     if not self.allow_measurement:
        #         ibm_estimator_zero_amplitude, ibm_estimator_zero_probability = self.ibm_estimator_executor.get_amplitude_of_zero(qc, qubit_count, shots=shots)

        # ── assemble results ───────────────────────────────────────────────────

        return Results(
            used_center_distance=used_center_distance,
            scaled_center_distance = scaled_center_distance,
            exact_result=exact_result,

            analitical_zero_amplitude=analitical_zero_amplitude,
            analitical_zero_probablity=analitical_zero_probablity,
            errors_for_analitical=sample_interpreter.get_errors(exact_result, analitical_zero_amplitude, analitical_zero_amplitude),
            
            sampled_zero_amplitude=sampled_zero_amplitude,
            sampled_zero_probability=sampled_zero_probability,
            errors_for_sampled=sample_interpreter.get_errors(exact_result, sampled_zero_amplitude, analitical_zero_amplitude),
            
            noisy_sampled_zero_amplitude=noisy_sampled_zero_amplitude,
            noisy_sampled_zero_probability=noisy_sampled_zero_probability,
            errors_for_noisy_sampled=sample_interpreter.get_errors(exact_result, noisy_sampled_zero_amplitude, analitical_zero_amplitude),
            
            estimator_zero_amplitude=estimator_zero_amplitude,
            estimator_zero_probability=estimator_zero_probability,
            errors_for_estimator=sample_interpreter.get_errors(exact_result, estimator_zero_amplitude, analitical_zero_amplitude),
            
            ibm_sampler_zero_amplitude=ibm_sampler_zero_amplitude,
            ibm_sampler_zero_probability=ibm_sampler_zero_probability,
            errors_for_ibm_sampler=sample_interpreter.get_errors(exact_result, ibm_sampler_zero_amplitude, analitical_zero_amplitude) if ibm_sampler_zero_amplitude is not None else None,
            
            ibm_estimator_zero_amplitude=ibm_estimator_zero_amplitude,
            ibm_estimator_zero_probability=ibm_estimator_zero_probability,
            errors_for_ibm_estimator=sample_interpreter.get_errors(exact_result, ibm_estimator_zero_amplitude, analitical_zero_amplitude) if ibm_estimator_zero_amplitude is not None else None,

            noisy_simulation_stats=noisy_simulation_stats,
            ibm_backend_stats=ibm_backend_stats,
        )
