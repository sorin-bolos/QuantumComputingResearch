import matplotlib.pyplot as plt
import numpy as np

from utils.dataclasses import Errors

class SampleInterpreter:

    def __init__(self):
        pass
    
    def plot_sampled_measurements(
        self,
        counts: dict,
        include_zero_values: bool = True,
        figsize: tuple = (12, 5),
        title: str = "Measurement Counts",
    ) -> plt.Figure:
        """Plot a measurement count histogram from pre-sampled counts.

        Parameters
        ----------
        counts : dict
            Mapping from basis-state bitstring to integer count, as returned
            by :func:`sample_measurement_counts`.
        include_zero_values : bool, optional
            When True (default) all basis states appear on the x-axis, even
            those with zero counts.  When False only observed states are shown.
        figsize : tuple, optional
            Matplotlib figure size in inches.
        title : str, optional
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not include_zero_values:
            counts = {k: v for k, v in counts.items() if v > 0}

        shots = sum(counts.values())
        states = sorted(counts.keys())
        values = [counts[s] for s in states]
        x = np.arange(len(states))

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(x, values, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"|{s}⟩" for s in states], rotation=90, fontsize=8)
        ax.set_xlabel("Basis state")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} ({shots} shots)")
        ax.set_xlim(-0.5, len(states) - 0.5)
        plt.tight_layout()
        return fig

    def get_zero_probability(self, counts: dict) -> float:
        """Return the probability of measuring the all-zero state |0…0⟩.

        Parameters
        ----------
        counts : dict
            Mapping from basis-state bitstring to integer count, as returned
            by :func:`sample_measurement_counts`.

        Returns
        -------
        float
            Probability estimate P(|0…0⟩) = count(|0…0⟩) / total_shots.
        """
        shots = sum(counts.values())
        if shots == 0:
            return 0.0
        zero_key = '0' * len(next(iter(counts)))
        return counts.get(zero_key, 0) / shots

    def get_zero_amplitude(self, counts: dict) -> float:
        """Return the square root of the all-zero state probability.

        For real non-negative wavefunctions this estimates the overlap amplitude
        ⟨0…0|ψ⟩ = √P(|0…0⟩).

        Parameters
        ----------
        counts : dict
            Mapping from basis-state bitstring to integer count.

        Returns
        -------
        float
        """
        return np.sqrt(self.get_zero_probability(counts))

    def print_measurement_counts(self, counts: dict, shots: int) -> None:
        """Print a text histogram of measurement counts.

        Parameters
        ----------
        counts : dict
            Mapping from basis-state bitstring to integer count,
            as returned by :func:`sample_measurement_counts`.
        shots : int
            Total number of shots (used to normalise the bar widths).
        """
        print(f"Measurement results ({shots} shots):")
        for state in sorted(counts.keys()):
            bar = '█' * int(counts[state] / shots * 40)
            print(f"  |{state}⟩ : {counts[state]:4d}  {bar}")

    def get_errors(self, theoretical: float, sampled: float, statevector_result: float):
        error_vs_continuous = abs(sampled - theoretical)
        percent_error_vs_continuous = error_vs_continuous/theoretical*100 if theoretical != 0 else 0.0

        if statevector_result:
            discretisation_error = abs(statevector_result  - theoretical)
            percent_discretisation_error = discretisation_error/theoretical*100 if theoretical != 0 else 0.0
            shot_noise = abs(sampled - statevector_result)
            percent_shot_noise = shot_noise/statevector_result*100 if statevector_result != 0 else 0.0

        return Errors(
            error_vs_continuous = error_vs_continuous,
            percent_error_vs_continuous = percent_error_vs_continuous,
            discretisation_error = discretisation_error,
            percent_discretisation_error = percent_discretisation_error,
            shot_noise = shot_noise,
            percent_shot_noise = percent_shot_noise
        )

    def print_errors(self, theoretical: float, sampled: float, statevector_result: float):
        print()
        print(f"Error vs continuous:  {abs(sampled - theoretical):.4f}  ({abs(sampled - theoretical)/theoretical*100:.2f} %)")
        if statevector_result:
            print(f"Discretisation error: {abs(statevector_result  - theoretical):.4f}  ({abs(statevector_result  - theoretical)/theoretical*100:.2f} %)")
            print(f"Shot noise:           {abs(sampled - statevector_result):.4f}  ({abs(sampled - statevector_result)/statevector_result*100:.2f} %)")

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
        x_marginal = psi.sum(axis=(1, 2))
        y_marginal = psi.sum(axis=(0, 2))
        z_marginal = psi.sum(axis=(0, 1))
        return x_marginal, y_marginal, z_marginal