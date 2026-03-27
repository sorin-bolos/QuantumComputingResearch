"""Reusable statevector visualisation helpers."""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator


def get_data_amplitudes(qc, n_qubits: int, sim=None) -> np.ndarray:
    """Run *qc* on a statevector simulator and return the renormalised data amplitudes.

    When mid-circuit measurements are present (ancilla-based uncompute), the full
    statevector is partitioned into ancilla-configuration blocks.  The dominant block
    is selected and renormalised to yield the pure data-qubit amplitudes.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to simulate (must not already have a save_statevector instruction).
    n_qubits : int
        Number of *data* qubits (the remaining qubits are treated as ancilla).
    sim : AerSimulator, optional
        Simulator instance to reuse.  A new ``AerSimulator(method='statevector')`` is
        created if not provided.

    Returns
    -------
    np.ndarray of complex
        1-D array of length ``2**n_qubits`` with the renormalised data amplitudes.
    """
    if sim is None:
        sim = AerSimulator(method='statevector')

    qc_measured = qc.copy()
    qc_measured.save_statevector()
    sv = np.array(sim.run(qc_measured, shots=1).result().get_statevector())

    n_anc = qc.num_qubits - n_qubits
    if n_anc > 0:
        sv_blocks = sv.reshape(2 ** n_anc, 2 ** n_qubits)
        block_norms = np.sum(np.abs(sv_blocks) ** 2, axis=1)
        anc_idx = np.argmax(block_norms)
        return sv_blocks[anc_idx] / np.sqrt(block_norms[anc_idx])
    return sv


def get_raw_amplitudes(qc, sim=None) -> np.ndarray:
    """Run *qc* on a statevector simulator and return the full statevector.

    Unlike :func:`get_data_amplitudes`, no ancilla partitioning or
    renormalisation is applied — the returned array covers all qubits.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to simulate (must not already have a save_statevector instruction).
    sim : AerSimulator, optional
        Simulator instance to reuse.

    Returns
    -------
    np.ndarray of complex
        1-D array of length ``2**qc.num_qubits`` with the raw statevector amplitudes.
    """
    if sim is None:
        sim = AerSimulator(method='statevector')

    qc_copy = qc.copy()
    qc_copy.save_statevector()
    return np.array(sim.run(qc_copy, shots=1).result().get_statevector())


def print_statevector(data_amps, n_qubits: int, threshold: float = 1e-6) -> None:
    """Print the non-negligible amplitudes of a statevector.

    Parameters
    ----------
    data_amps : array-like of complex
        1-D array of length ``2**n_qubits``.
    n_qubits : int
        Number of qubits (determines basis-state labels).
    threshold : float, optional
        Amplitudes with ``|amp| <= threshold`` are suppressed.
    """
    data_amps = np.asarray(data_amps)
    print("Statevector amplitudes (|basis⟩ : amplitude):")
    for i, amp in enumerate(data_amps):
        if abs(amp) > threshold:
            basis = format(i, f"0{n_qubits}b")
            print(f"  |{basis}⟩ : {amp:.4f}  (prob = {abs(amp) ** 2:.4f})")


def plot_statevector_real_imag(
    data_amps,
    n_qubits: int,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot a statevector as two stacked vertical bar charts: real on top, imaginary below.

    Parameters
    ----------
    data_amps : array-like of complex
        1-D array of length 2**n_qubits with the statevector amplitudes.
    n_qubits : int
        Number of qubits (determines the basis-state labels).
    title : str, optional
        Figure suptitle.
    figsize : tuple, optional
        Matplotlib figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (also displayed via plt.show()).
    """
    title="Statevector – Real-Img"
    data_amps = np.asarray(data_amps)
    n_states = 2 ** n_qubits
    basis_labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    real_parts = data_amps.real
    imag_parts = data_amps.imag

    x = np.arange(n_states)
    tick_labels = [f"|{b}⟩" for b in basis_labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    ax1.bar(x, real_parts, color="steelblue", edgecolor="white", linewidth=0.5)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Real Amplitudes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(-0.5, n_states - 0.5)

    ax2.bar(x, imag_parts, color="coral", edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Imaginary Amplitudes")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax2.set_ylabel("Amplitude")
    ax2.set_xlim(-0.5, n_states - 0.5)
    plt.tight_layout()
    return fig


def plot_statevector_modulus_phase(
    data_amps,
    n_qubits: int,
    title: str = "Statevector – Modulus & Phase",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot a statevector as two stacked vertical bar charts: modulus on top, phase below.

    Parameters
    ----------
    data_amps : array-like of complex
        1-D array of length 2**n_qubits with the statevector amplitudes.
    n_qubits : int
        Number of qubits (determines the basis-state labels).
    title : str, optional
        Figure suptitle.
    figsize : tuple, optional
        Matplotlib figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object, ready to be displayed by the caller.
    """
    data_amps = np.asarray(data_amps)
    n_states = 2 ** n_qubits
    basis_labels = [format(i, f"0{n_qubits}b") for i in range(n_states)]
    modulus = np.abs(data_amps)
    phase = np.angle(data_amps)   # radians in (-π, π]

    x = np.arange(n_states)
    tick_labels = [f"|{b}⟩" for b in basis_labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    ax1.bar(x, modulus, color="steelblue", edgecolor="white", linewidth=0.5)
    ax1.set_title("Modulus  |α|")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax1.set_ylabel("|Amplitude|")
    ax1.set_xlim(-0.5, n_states - 0.5)
    ax1.set_ylim(bottom=0)

    ax2.bar(x, phase, color="darkorange", edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Phase  arg(α)  [radians]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax2.set_ylabel("Phase (rad)")
    ax2.set_xlim(-0.5, n_states - 0.5)
    ax2.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax2.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

    plt.tight_layout()
    return fig


def sample_measurement_counts(qc, n_qubits: int, shots: int = 1024) -> dict:
    """Run *qc* with shot-based measurements and return per-data-qubit counts.

    A dedicated classical register is added to the circuit so that the data-qubit
    results can be isolated from any ancilla / mid-circuit classical registers that
    Aer includes in its space-separated bitstrings.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to sample (must not already have a final measurement).
    n_qubits : int
        Number of data qubits to measure (qubits 0 … n_qubits-1).
    shots : int, optional
        Number of measurement shots (default 1024).

    Returns
    -------
    dict
        Mapping from binary string (MSB first, length *n_qubits*) to integer count.
        All ``2**n_qubits`` basis states are present; unobserved states have count 0.
    """
    sim = AerSimulator(method='statevector')

    qc_sample = qc.copy()
    result_reg = ClassicalRegister(n_qubits, 'result')
    qc_sample.add_register(result_reg)
    qc_sample.measure(list(range(n_qubits)), result_reg)

    raw_counts = sim.run(qc_sample, shots=shots).result().get_counts()

    # Bitstring format with multiple registers: '<result_bits> <mid_bit>'
    # Qiskit prints registers right-to-left in creation order, so the last-added
    # register ('result') is the leftmost space-separated token.
    counts: dict = {}
    for bitstring, count in raw_counts.items():
        data_bits = bitstring.split()[0]
        counts[data_bits] = counts.get(data_bits, 0) + count

    # Ensure all basis states are present (zeros for unobserved states)
    all_basis = [format(i, f'0{n_qubits}b') for i in range(2 ** n_qubits)]
    return {b: counts.get(b, 0) for b in all_basis}


def sample_raw_measurement_counts(qc, shots: int = 1024) -> dict:
    """Run *qc* with shot-based measurements on all qubits and return raw counts.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to sample (must not already have a final measurement).
    shots : int, optional
        Number of measurement shots (default 1024).

    Returns
    -------
    dict
        Mapping from binary string (MSB first, length = qc.num_qubits) to
        integer count. All observed basis states are present.
    """
    sim = AerSimulator(method='statevector')

    qc_sample = qc.copy()
    result_reg = ClassicalRegister(qc.num_qubits, 'result')
    qc_sample.add_register(result_reg)
    qc_sample.measure(list(range(qc.num_qubits)), result_reg)

    raw_counts = sim.run(qc_sample, shots=shots).result().get_counts()

    counts = {}
    for bitstring, count in raw_counts.items():
        data_bits = bitstring.split()[0]
        counts[data_bits] = counts.get(data_bits, 0) + count

    return counts


def get_ancilla_amplitudes(qc, n_data: int, sim=None) -> dict:
    """Simulate *qc* and return the probability of each ancilla basis state.

    The statevector is reshaped into blocks of shape
    ``(2**n_anc, 2**n_data)``.  Each row is one ancilla basis state; the
    probability of that ancilla outcome is the squared norm of its row.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to simulate.
    n_data : int
        Number of data qubits (qubits 0 … n_data-1).  The remaining qubits
        are treated as ancilla.
    sim : AerSimulator, optional
        Simulator instance to reuse.

    Returns
    -------
    dict
        Mapping from ancilla basis-state bitstring (MSB first) to probability.
        Only states with non-negligible probability (> 1e-10) are included.
    """
    if sim is None:
        sim = AerSimulator(method='statevector')

    n_anc = qc.num_qubits - n_data
    if n_anc <= 0:
        return {'': 1.0}

    qc_copy = qc.copy()
    qc_copy.save_statevector()
    sv = np.array(sim.run(qc_copy, shots=1).result().get_statevector())

    sv_blocks = sv.reshape(2 ** n_anc, 2 ** n_data)
    anc_probs = np.sum(np.abs(sv_blocks) ** 2, axis=1)

    return {
        format(i, f'0{n_anc}b'): float(prob)
        for i, prob in enumerate(anc_probs)
        if prob > 1e-10
    }


def print_measurement_counts(counts: dict, shots: int) -> None:
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
