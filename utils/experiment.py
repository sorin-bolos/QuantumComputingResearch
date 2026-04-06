from utils.integrals import Integals
from utils.simulation_excutor import SimulationExecutor
from utils.sample_interpreter import SampleInterpreter
from utils.noisy_sampler_executor import NoisySamplerExecutor
from utils.noisy_estimator_executor import NoisyEstimatorExecutor
from utils.ibm_sampler_executor import IbmSamplerExecutor
from utils.ibm_estimator_executor import IbmEstimatorExecutor
from utils.dataclasses import IntegralContext, Results, CircuitStats, Errors, RunResults, SimulationResults
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
        
        integrals = Integals(self.allow_measurement, self.optimize_t_gates)

        used_center_distance = scaled_center_distance / scale
        exact_result = integrals.get_1s_1d_overlap_exact_result(decay_constant, used_center_distance)

        context = IntegralContext(
            used_center_distance=used_center_distance,
            scaled_center_distance=scaled_center_distance,
            exact_result=exact_result,
        )

        qc = integrals.get_s1_1d_overlap_circuit(qubit_count, scaled_decay_constant, scaled_center_distance)

        stats = self._get_circuit_stats(qc)
        raw_results = self._run_all_methods(qc, qubit_count, shots)
        
        simulation_results = []
        analitical_result = None
        for run_result in raw_results:
            if run_result.run_name == "Analytical (statevector)":
                analitical_result = run_result.run_result

            errors = self._get_errors(context.exact_result, run_result.run_result, analitical_result)
            simulation_results.append(SimulationResults(
                name=run_result.run_name,
                result=run_result.run_result,
                errors=errors
            ))

        return Results(
            context=context,
            stats=stats,
            results=simulation_results
        )
    
    def run_single_s1_1d_kinetic_integral(
            self,
            qubit_count: int,
            decay_constant: float,
            max_range: int, # the size of the space that will be represented by 2^qubit_count values
            center_distance: float, # the distance between integral centers
            shots: int = 1024) -> Results:

        scale = (2 ** qubit_count) / max_range
        scaled_center_distance = round(center_distance * scale)
        scaled_decay_constant = decay_constant / scale
        
        integrals = Integals(self.allow_measurement, self.optimize_t_gates)

        used_center_distance = scaled_center_distance / scale
        exact_result = integrals.get_1s_1d_kinetic_exact_result(decay_constant, used_center_distance)

        context = IntegralContext(
            used_center_distance=used_center_distance,
            scaled_center_distance=scaled_center_distance,
            exact_result=exact_result,
        )

        qc = integrals.get_s1_1d_kinetic_derivative(qubit_count, scaled_decay_constant, scaled_center_distance)

        stats = self._get_circuit_stats(qc)
        raw_results = self._run_all_methods(qc, qubit_count, shots)
        
        simulation_results = []
        analitical_result = None
        for run_result in raw_results:
            result = -0.5* decay_constant**2 * run_result.run_result

            if run_result.run_name == "Analytical (statevector)":
                analitical_result = result

            errors = self._get_errors(context.exact_result, result, analitical_result)
            simulation_results.append(SimulationResults(
                name=run_result.run_name,
                result=result,
                errors=errors
            ))

        return Results(
            context=context,
            stats=stats,
            results=simulation_results
        )

    def run_single_s1_1d_kinetic_analytical(
            self,
            qubit_count: int,
            decay_constant: float,
            max_range: int,
            center_distance: float,
            shots: int = 1024) -> Results:

        scale = (2 ** qubit_count) / max_range
        scaled_center_distance = round(center_distance * scale)
        scaled_decay_constant = decay_constant / scale

        used_center_distance = scaled_center_distance / scale

        integrals = Integals(self.allow_measurement, self.optimize_t_gates)

        # Exact continuous result: T₁₂ = ½a²(1 - a|R|)e^(-a|R|)
        exact_result = integrals.get_1s_1d_kinetic_exact_result(decay_constant, used_center_distance)

        context = IntegralContext(
            used_center_distance=used_center_distance,
            scaled_center_distance=scaled_center_distance,
            exact_result=exact_result,
        )

        
        qc = integrals.get_s1_1d_overlap_circuit(qubit_count, scaled_decay_constant, scaled_center_distance)

        stats = self._get_circuit_stats(qc)
        raw_results = self._run_all_methods(qc, qubit_count, shots)

        kinetic_scale  = -0.5 * decay_constant ** 2
        kinetic_offset =  decay_constant * np.exp(-decay_constant * abs(used_center_distance))

        simulation_results = []
        analitical_result = None
        for run_result in raw_results:
            result = kinetic_scale * run_result.run_result + kinetic_offset

            if run_result.run_name == "Analytical (statevector)":
                analitical_result = result

            errors = self._get_errors(context.exact_result, result, analitical_result)
            simulation_results.append(SimulationResults(
                name=run_result.run_name,
                result=result,
                errors=errors
            ))

        return Results(
            context=context,
            stats=stats,
            results=simulation_results
        )
    
    def run_single_s1_1d_kinetic_finite_diff(
            self,
            qubit_count: int,
            decay_constant: float,
            max_range: int,
            center_distance: float,
            shots: int = 1024) -> Results:
        
        scale = (2 ** qubit_count) / max_range
        scaled_center_distance = round(center_distance * scale)
        scaled_decay_constant = decay_constant / scale

        used_center_distance = scaled_center_distance / scale

        integrals = Integals(self.allow_measurement, self.optimize_t_gates)

        # Exact continuous result: T₁₂ = ½a²(1 - a|R|)e^(-a|R|)
        exact_result = integrals.get_1s_1d_kinetic_exact_result(decay_constant, used_center_distance)

        context = IntegralContext(
            used_center_distance=used_center_distance,
            scaled_center_distance=scaled_center_distance,
            exact_result=exact_result,
        )

        qc_0 = integrals.get_s1_1d_overlap_circuit(qubit_count, scaled_decay_constant, scaled_center_distance)
        qc_1 = integrals.get_s1_1d_overlap_circuit(qubit_count, scaled_decay_constant, scaled_center_distance + 1)
        qc_m1 = integrals.get_s1_1d_overlap_circuit(qubit_count, scaled_decay_constant, scaled_center_distance - 1)

        stats = self._get_circuit_stats(qc_0)
        raw_results_0 = self._run_all_methods(qc_0, qubit_count, shots)
        raw_results_1 = self._run_all_methods(qc_1, qubit_count, shots)
        raw_results_m1 = self._run_all_methods(qc_m1, qubit_count, shots)

        simulation_results = []
        analitical_result = None
        scale_squared = scale ** 2
        for i in range(len(raw_results_0)):
            result = (raw_results_0[i].run_result - 0.5 * raw_results_1[i].run_result - 0.5 * raw_results_m1[i].run_result)

            if i == 0:
                analitical_result = result

            errors = self._get_errors(context.exact_result, result, analitical_result)
            simulation_results.append(SimulationResults(
                name=raw_results_0[i].run_name,
                result=result,
                errors=errors
            ))

        return Results(
            context=context,
            stats=stats,
            results=simulation_results
        )
    
    def run_single_V11A_1D_integral(
            self,
            qubit_count: int,
            decay_constant: float,
            Z: float,
            max_range: int, # the size of the space that will be represented by 2^qubit_count values
            shots: int = 1024) -> Results:

        scale = (2 ** qubit_count) / max_range
        scaled_decay_constant = decay_constant / scale
        
        integrals = Integals(self.allow_measurement, self.optimize_t_gates)

        exact_result = integrals.get_V11B_1D_exact(decay_constant, 1.375)

        context = IntegralContext(
            used_center_distance=None,
            scaled_center_distance=None,
            exact_result=exact_result,
        )

        qc = integrals.get_V11B_1D_circuit(qubit_count, scaled_decay_constant, decay_constant, max_range)

        stats = self._get_circuit_stats(qc)
        raw_results = self._run_all_methods(qc, qubit_count, shots, run_noisy_estimation=False, run_noisy_simulation=False)
        
        simulation_results = []
        analitical_result = None
        for run_result in raw_results:
            result = -1 * Z * run_result.run_result

            if run_result.run_name == "Analytical (statevector)":
                analitical_result = result

            errors = self._get_errors(context.exact_result, result, analitical_result)
            simulation_results.append(SimulationResults(
                name=run_result.run_name,
                result=result,
                errors=errors
            ))

        return Results(
            context=context,
            stats=stats,
            results=simulation_results
        )
    
    def _get_circuit_stats(self, qc) -> list[CircuitStats]:
        stats = [self.resource_estimator.get_circuit_stats(qc, self.fake_backend)]
        if self.ibm_backend is not None:
            stats.append(self.resource_estimator.get_circuit_stats(qc, self.ibm_backend))
        return stats
    
    def _simulate_statevector(self, qc, qubit_count):
        simulation_executor = SimulationExecutor()
        data_amps = simulation_executor.get_data_amplitudes(qc, qubit_count)
        analitical_zero_amplitude = simulation_executor.get_analytical_zero_amplitude(data_amps)
        return RunResults(
            run_name="Analytical (statevector)",
            run_result=analitical_zero_amplitude
        )

    def _simulate_counts(self, qc, qubit_count, shots):
        simulation_executor = SimulationExecutor()
        counts = simulation_executor.sample_measurement_counts(qc, qubit_count, shots=shots)
        sample_interpreter = SampleInterpreter()
        sampled_zero_amplitude = sample_interpreter.get_zero_amplitude(counts)
        return RunResults(
            run_name="Sampled (counts)",
            run_result=sampled_zero_amplitude
        )

    def _noisy_simulation(self, qc, qubit_count, shots):
        counts = self._noisy_sampler_executor.sample_measurement_counts(qc, qubit_count, shots=shots)
        sample_interpreter = SampleInterpreter()
        noisy_sampled_zero_amplitude = sample_interpreter.get_zero_amplitude(counts)
        return RunResults(
            run_name="Noisy (simulation)",
            run_result=noisy_sampled_zero_amplitude
        )

    def _noisy_estimation(self, qc, qubit_count, shots):
        estimator_zero_amplitude, _ = self._noisy_estimator_executor.get_amplitude_of_zero(qc, qubit_count, shots=shots)
        return RunResults(
            run_name="Noisy (estimation)",
            run_result=estimator_zero_amplitude
        )

    def _ibm_sampling(self, qc, qubit_count, shots):
        if self.ibm_sampler_executor is None:
            return None
        ibm_counts = self.ibm_sampler_executor.sample_measurement_counts(qc, qubit_count, shots=shots)
        sample_interpreter = SampleInterpreter()
        ibm_sampler_zero_amplitude = sample_interpreter.get_zero_amplitude(ibm_counts)
        return RunResults(
            run_name="IBM (sampling)",
            run_result=ibm_sampler_zero_amplitude
        )
    
    def _ibm_estimation(self, qc, qubit_count, shots):
        if self.ibm_estimator_executor is None:
            return None
        ibm_estimator_zero_amplitude, _ = self.ibm_estimator_executor.get_amplitude_of_zero(qc, qubit_count, shots=shots)
        return RunResults(
            run_name="IBM (estimation)",
            run_result=ibm_estimator_zero_amplitude
        )
    
    def _run_all_methods(self, qc, qubit_count, shots,
                         run_noisy_simulation=True, 
                         run_noisy_estimation=True, 
                         run_ibm_sampling=True, 
                         run_ibm_estimation=True) -> list[RunResults]:
        
        analitical_zero_amplitude = self._simulate_statevector(qc, qubit_count)

        sampled_zero_amplitude = self._simulate_counts(qc, qubit_count, shots)
        
        noisy_sampled_zero_amplitude = None
        if run_noisy_simulation:
            noisy_sampled_zero_amplitude = self._noisy_simulation(qc, qubit_count, shots)
        
        estimator_zero_amplitude = None
        if run_noisy_estimation:
            estimator_zero_amplitude = self._noisy_estimation(qc, qubit_count, shots)
        
        ibm_sampler_zero_amplitude = None
        if run_ibm_sampling and self.ibm_backend is not None:
            ibm_sampler_zero_amplitude = self._ibm_sampling(qc, qubit_count, shots)
        
        ibm_estimator_zero_amplitude = None
        if run_ibm_estimation and self.ibm_backend is not None and not self.allow_measurement:
            ibm_estimator_zero_amplitude = self._ibm_estimation(qc, qubit_count, shots)

        results = [
            analitical_zero_amplitude,
            sampled_zero_amplitude,
            noisy_sampled_zero_amplitude,
            estimator_zero_amplitude,
            ibm_sampler_zero_amplitude,
            ibm_estimator_zero_amplitude
        ]
        return [r for r in results if r is not None]

    def _get_errors(self, exact_result: float, run_result: float, analitical_result: float) -> Errors:
        return SampleInterpreter().get_errors(exact_result, run_result, analitical_result)

            
