from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister

from utils.logic import LogicalOperator

class ArithmeticOperator:
    """Arithmetic operators for quantum circuits."""
    def __init__(self, circuit: QuantumCircuit, allow_measurement: bool = True, optimize_t_gates: bool =True):
        self.circuit = circuit
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates

    def add_constant(self, qubit_count: int, constant: int):

        constant = constant % (2**qubit_count)

        add_circuit = self._add_constant(qubit_count, constant)
        n_total = add_circuit.num_qubits  # data + ancilla
        combined = QuantumCircuit(n_total)
        for creg in add_circuit.cregs:
            combined.add_register(creg)
        combined.compose(self.circuit, qubits=range(qubit_count), inplace=True)
        combined.compose(add_circuit, qubits=range(n_total),
                         clbits=list(combined.clbits), inplace=True)
        return combined

    def subtract_constant(self, qubit_count: int, constant: int):
        constant = constant % (2**qubit_count)
        constant = (2**qubit_count) - constant
        
        return self.add_constant(qubit_count, constant)

    @staticmethod
    def _add_classically_controlled_X(qc: QuantumCircuit, qubits: list, bit: bool):
        if bit:
            for q in qubits:
                qc.x(q)

    @staticmethod
    def _int_to_bits(const: int, num_bits: int = None):
        bits = [(const >> i) & 1 for i in range(num_bits or const.bit_length())]
        return bits  # LSB first
    
    @staticmethod
    def _remove_leading_0s(bits: list):
        first_one = next((i for i, b in enumerate(bits) if b), len(bits))
        return bits[first_one:]

    def _add_constant_clean(self, constant_bits: list, qubit_count: int):
        data = QuantumRegister(qubit_count, 'data')
        anc  = AncillaRegister(qubit_count - 3, 'anc')
        qc    = QuantumCircuit(data, anc, name=f"+{int(''.join(map(str, constant_bits)), 2)}")

        logical_op = LogicalOperator(qc, self.allow_measurement, self.optimize_t_gates)

        self._add_classically_controlled_X(qc, [data[0], data[1]], constant_bits[1])
        logical_op.apply_temporary_and(data[0], data[1], anc[0])

        self._add_classically_controlled_X(qc, [anc[0]], constant_bits[1]^constant_bits[2])
        self._add_classically_controlled_X(qc, [data[2]], constant_bits[2])
        logical_op.apply_temporary_and(anc[0], data[2], anc[1])

        self._add_classically_controlled_X(qc, [anc[1]], constant_bits[2]^constant_bits[3])
        self._add_classically_controlled_X(qc, [data[3]], constant_bits[3])
        logical_op.apply_temporary_and(anc[1], data[3], anc[2])

        self._add_classically_controlled_X(qc, [anc[2]], constant_bits[3]^constant_bits[4])
        self._add_classically_controlled_X(qc, [data[4]], constant_bits[4])
        logical_op.apply_temporary_and(anc[2], data[4], anc[3])

        self._add_classically_controlled_X(qc, [anc[3]], constant_bits[4]^constant_bits[5])
        self._add_classically_controlled_X(qc, [data[5]], constant_bits[5])
        qc.ccx(anc[3], data[5], data[6])

        self._add_classically_controlled_X(qc, [anc[3]], constant_bits[5])

        qc.cx(anc[3], data[5])
        self._add_classically_controlled_X(qc, [anc[3]], constant_bits[4])

        logical_op.uncompute_temporary_and(anc[2], data[4], anc[3])
        self._add_classically_controlled_X(qc, [anc[2]], constant_bits[4])

        qc.cx(anc[2], data[4])
        self._add_classically_controlled_X(qc, [anc[2]], constant_bits[3])

        logical_op.uncompute_temporary_and(anc[1], data[3], anc[2])
        self._add_classically_controlled_X(qc, [anc[1]], constant_bits[3])

        qc.cx(anc[1], data[3])
        self._add_classically_controlled_X(qc, [anc[1]], constant_bits[2])

        logical_op.uncompute_temporary_and(anc[0], data[2], anc[1])
        self._add_classically_controlled_X(qc, [anc[0]], constant_bits[2])

        qc.cx(anc[0], data[2])
        self._add_classically_controlled_X(qc, [anc[0]], constant_bits[1])

        logical_op.uncompute_temporary_and(data[0], data[1], anc[0])
        self._add_classically_controlled_X(qc, [data[0]], constant_bits[0])

        qc.cx(data[0], data[1])

        qc.x(data[0])
        self._add_classically_controlled_X(qc, [data[6]], constant_bits[5]^constant_bits[6])

        return qc
    
    def _add_constant(self, qubit_count, constant):
        if constant == 0:
            return QuantumCircuit(qubit_count)
        
        constant_bits = self._int_to_bits(constant, num_bits=qubit_count)
        constant_bits = self._remove_leading_0s(constant_bits)
        bit_count = len(constant_bits)
        offset = qubit_count - bit_count  # number of new LSB qubits to prepend

        # optimization
        if bit_count == 1:
            qc = QuantumCircuit(qubit_count)
            qc.x(qubit_count-1)
            return qc

        adderCircuit = self._add_constant_clean(constant_bits, bit_count)

        # Build expanded circuit: full-width data register + same ancilla register
        new_data  = QuantumRegister(qubit_count, 'data')
        anc       = AncillaRegister(bit_count - 3, 'anc')
        qubit_map = [new_data[offset + i] for i in range(bit_count)] + list(anc)

        if self.allow_measurement and self.optimize_t_gates:
            # Carry the single mid-circuit classical bit into the result circuit.
            # It lives in its own named register so it never collides with any
            # classical register added later (e.g. a final measurement register).
            mid    = ClassicalRegister(1, 'mid')
            result = QuantumCircuit(new_data, anc, mid, name=f"+{int(''.join(map(str, constant_bits)), 2)}")
            result.compose(adderCircuit, qubits=qubit_map, clbits=[mid[0]], inplace=True)
        else:
            result = QuantumCircuit(new_data, anc, name=f"+{int(''.join(map(str, constant_bits)), 2)}")
            result.compose(adderCircuit, qubits=qubit_map, inplace=True)

        return result
    
    def _add_constant_qft(self, qubit_count: int, constant: int) -> QuantumCircuit:
        from qiskit.circuit.library import QFT
        import numpy as np

        """Returns a circuit that maps |x⟩ → |(x + c) mod 2^n⟩."""
        qc = QuantumCircuit(qubit_count, name=f"+{constant}")
        qc.append(QFT(qubit_count, do_swaps=False), range(qubit_count))
        for k in range(qubit_count):
            angle = 2 * np.pi * constant / (2 ** (k + 1))
            qc.p(angle, k)          # P(θ)|0⟩=|0⟩, P(θ)|1⟩=e^{iθ}|1⟩

        qc.append(QFT(qubit_count, do_swaps=False, inverse=True), range(qubit_count))

        return qc

    def _add_constant_qft_dagger(self, qubit_count: int, constant: int) -> QuantumCircuit:
        from qiskit.circuit.library import QFT
        import numpy as np

        qc = QuantumCircuit(qubit_count, name=f"+{constant}")
        
        qc.append(QFT(qubit_count, do_swaps=False), range(qubit_count))
        
        for k in range(qubit_count):
            angle = -2 * np.pi * constant / (2 ** (k + 1))
            qc.p(angle, k)          # P(θ)|0⟩=|0⟩, P(θ)|1⟩=e^{iθ}|1⟩

        qc.append(QFT(qubit_count, do_swaps=False, inverse=True), range(qubit_count))
        
        return qc