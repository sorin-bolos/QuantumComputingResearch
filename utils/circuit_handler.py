from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister

class CircuitHandler:

    def __init__(self):
        pass

    @staticmethod
    def concatenate(circuit1: QuantumCircuit, circuit2: QuantumCircuit) -> QuantumCircuit:
        total_qubits = max(circuit1.num_qubits, circuit2.num_qubits)
        combined = QuantumCircuit(total_qubits)

        # Add circuit1's classical registers directly — same Clbit objects,
        # so no remapping is needed when composing circuit1.
        for creg in circuit1.cregs:
            combined.add_register(creg)

        # Add circuit2's classical registers, renaming any name conflicts.
        c2_clbit_map = {}
        existing_names = {creg.name for creg in combined.cregs}
        for creg in circuit2.cregs:
            name = creg.name
            suffix = 2
            while name in existing_names:
                name = f"{creg.name}_{suffix}"
                suffix += 1
            new_reg = ClassicalRegister(creg.size, name)
            combined.add_register(new_reg)
            existing_names.add(name)
            for orig_bit, new_bit in zip(creg, new_reg):
                c2_clbit_map[orig_bit] = new_bit

        combined.compose(
            circuit1,
            qubits=range(circuit1.num_qubits),
            clbits=list(circuit1.clbits),
            inplace=True,
        )
        combined.compose(
            circuit2,
            qubits=range(circuit2.num_qubits),
            clbits=[c2_clbit_map[b] for b in circuit2.clbits],
            inplace=True,
        )

        return combined
