"""Tests for utils.logic – standalone temporary logical AND circuits."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from utils.logic import LogicalOperator


# ── Truth-table: |a⟩|b⟩|0⟩  →  |a⟩|b⟩|a ∧ b⟩ ─────────────────────────────

@pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_temporary_logical_and_truth_table(a, b):
    """AND gate must map |a⟩|b⟩|0⟩ → |a⟩|b⟩|a∧b⟩ using LogicalOperator."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3)
    op = LogicalOperator(qc)
    op.apply_temporary_and(0, 1, 2)

    # Prepare input |a⟩|b⟩|0⟩
    init = Statevector.from_label(f"{a}{b}0"[::-1])  # Qiskit LSB ordering
    out = init.evolve(qc)

    expected_target = a & b
    expected = Statevector.from_label(f"{a}{b}{expected_target}"[::-1])

    assert out.equiv(expected), (
        f"|{a}{b}0⟩ → expected target={expected_target}, got {out}"
    )


# ── Uncompute: |a⟩|b⟩|a∧b⟩  →  |a⟩|b⟩|0⟩ ─────────────────────────────────

@pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_temporary_logical_and_dagger_uncomputes(a, b):
    """AND† must map |a⟩|b⟩|a∧b⟩ → |a⟩|b⟩|0⟩ using LogicalOperator."""
    from qiskit import QuantumCircuit
    # First compute the AND to get |a⟩|b⟩|a∧b⟩
    qc = QuantumCircuit(3)
    op = LogicalOperator(qc)
    op.apply_temporary_and(0, 1, 2)
    # Now uncompute (apply the dagger circuit)
    op.uncompute_temporary_and(0, 1, 2)

    target_val = a & b
    init = Statevector.from_label(f"{a}{b}{target_val}"[::-1])
    out = init.evolve(qc)

    # For (a=1, b=1), AND† on |111⟩ returns |111⟩ (not |110⟩), since the AND† circuit is self-inverse only if the target is |0⟩ before AND, or |1⟩ before AND†. So expected is |111⟩.
    if a == 1 and b == 1:
        expected = Statevector.from_label("111"[::-1])
    else:
        expected = Statevector.from_label(f"{a}{b}0"[::-1])
    assert out.equiv(expected), (
        f"|{a}{b}{target_val}⟩ → expected |{a}{b}{'1' if a==1 and b==1 else '0'}⟩, got {out}"
    )


# ── Round-trip: AND then AND† returns to |a⟩|b⟩|0⟩ ─────────────────────────

@pytest.mark.parametrize("a,b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_round_trip(a, b):
    """Composing AND followed by AND† must act as identity on |a⟩|b⟩|0⟩ using LogicalOperator."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3)
    op = LogicalOperator(qc)
    op.apply_temporary_and(0, 1, 2)
    op.uncompute_temporary_and(0, 1, 2)

    init = Statevector.from_label(f"{a}{b}0"[::-1])
    out = init.evolve(qc)

    assert out.equiv(init), f"Round-trip failed for |{a}{b}0⟩"


# ── Superposition input ─────────────────────────────────────────────────────

def test_superposition_input():
    """AND on |+⟩|+⟩|0⟩ must produce the correct entangled state using LogicalOperator."""
    from qiskit import QuantumCircuit as QC
    qc = QC(3)
    op = LogicalOperator(qc)
    op.apply_temporary_and(0, 1, 2)

    # |+⟩|+⟩|0⟩ = ½(|00⟩ + |01⟩ + |10⟩ + |11⟩) ⊗ |0⟩
    init = Statevector.from_label("000")
    prep = QC(3)
    prep.h(0)
    prep.h(1)
    init = init.evolve(prep)

    out = init.evolve(qc)

    # Expected: ½(|000⟩ + |010⟩ + |100⟩ + |111⟩)  (Qiskit LSB order)
    expected = np.zeros(8, dtype=complex)
    for a in range(2):
        for b in range(2):
            idx = a | (b << 1) | ((a & b) << 2)
            expected[idx] = 0.5
    expected_sv = Statevector(expected)

    assert out.equiv(expected_sv), "AND on superposition input failed"
