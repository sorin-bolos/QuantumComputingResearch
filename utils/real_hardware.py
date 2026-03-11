"""Utilities for analysing quantum circuits against real IBM hardware constraints."""

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeFez


_FAKE_FEZ = None


def _get_fake_fez():
    global _FAKE_FEZ
    if _FAKE_FEZ is None:
        _FAKE_FEZ = FakeFez()
    return _FAKE_FEZ


def analyse_for_ibm_fez(qc: QuantumCircuit, optimization_level: int = 2) -> dict:
    """Transpile *qc* to IBM Fez constraints and report hardware-relevant metrics.

    IBM Fez is a 156-qubit Heron R2 processor with CZ as its native
    two-qubit gate and heavy-hex connectivity.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to analyse (arbitrary gate set / connectivity).
    optimization_level : int
        Qiskit transpiler optimization level (0-3).

    Returns
    -------
    dict with keys:
        original_depth        – depth of the input circuit
        original_num_qubits   – number of qubits in the input circuit
        transpiled_depth      – depth after transpilation to ibm_fez
        num_cz_gates          – number of CZ (two-qubit) gates
        num_1q_gates          – number of single-qubit gates
        total_gates           – total gate count
        gate_counts           – dict of gate-type -> count
        transpiled_circuit    – the transpiled QuantumCircuit object
    """
    backend = _get_fake_fez()

    qc_t = transpile(
        qc,
        backend=backend,
        optimization_level=optimization_level,
    )

    ops = qc_t.count_ops()
    num_cz = ops.get("cz", 0)
    num_1q = sum(v for k, v in ops.items() if k in ("sx", "rz", "x", "id"))

    result = {
        "original_depth": qc.depth(),
        "original_num_qubits": qc.num_qubits,
        "transpiled_depth": qc_t.depth(),
        "num_cz_gates": num_cz,
        "num_1q_gates": num_1q,
        "total_gates": sum(ops.values()),
        "gate_counts": dict(ops),
        "transpiled_circuit": qc_t,
    }

    return result


def print_ibm_fez_analysis(qc: QuantumCircuit, optimization_level: int = 2) -> dict:
    """Transpile, print a summary, and return the analysis dict."""
    r = analyse_for_ibm_fez(qc, optimization_level)

    print("=" * 55)
    print("  IBM Fez (156-qubit Heron R2) – Circuit Analysis")
    print("=" * 55)
    print(f"  Original circuit:  {r['original_num_qubits']} qubits,  "
          f"depth {r['original_depth']}")
    print(f"  Transpiled circuit: depth {r['transpiled_depth']}")
    print(f"  Two-qubit (CZ) gates:  {r['num_cz_gates']}")
    print(f"  Single-qubit gates:    {r['num_1q_gates']}")
    print(f"  Total gates:           {r['total_gates']}")
    print(f"  Gate breakdown:        {r['gate_counts']}")
    print("=" * 55)

    return r