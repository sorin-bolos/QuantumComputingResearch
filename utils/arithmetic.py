from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister

from utils.logic import LogicalOperator

class ArithmeticOperator:
    """Arithmetic operators for quantum circuits."""
    def __init__(self, circuit: QuantumCircuit, allow_measurement: bool = True, optimize_t_gates: bool =True):
        self.circuit = circuit
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates

    def add_constant(self, qubit_count: int, constant: int):
        

        # constant = constant % (2**qubit_count)

        # add_circuit = self._add_constant(qubit_count, constant)
        add_circuit = self.m1(constant, qubit_count)

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
    def _int_to_bits(const: int, num_bits: int = None):
        bits = [(const >> i) & 1 for i in range(num_bits or const.bit_length())]
        return bits  # LSB first
    
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
    
    def _add_offset(self, qc: QuantumCircuit, offset: int) -> QuantumCircuit:
        """Return a new circuit with *offset* identity qubits prepended at the LSB.

        The original circuit's gates are shifted up by *offset* qubit positions.
        Classical registers and ancilla registers are preserved unchanged.
        """
        if offset == 0:
            return qc

        from qiskit.circuit import Qubit
        new_qc = QuantumCircuit()
        new_qc.add_bits([Qubit() for _ in range(offset)])  # anonymous LSB qubits, no named register

        for qreg in qc.qregs:           # shift original qubits to offset..offset+n-1
            new_qc.add_register(qreg)
        for creg in qc.cregs:           # preserve classical registers as-is
            new_qc.add_register(creg)

        new_qc.compose(
            qc,
            qubits=list(qc.qubits),     # same qubit objects, now at offset positions
            clbits=list(qc.clbits),
            inplace=True,
        )
        return new_qc

    def _constant_to_range(self, constant: int, qubit_count: int):
        return constant % (2**qubit_count)
    
    def _remove_zero_lsbs(self, constant: int):
        offset = 0
        while constant % 2 == 0:
            offset += 1
            constant = constant // 2
        
        return constant, offset
    
    def _classic_cx(self, qc: QuantumCircuit, qubit: int, bit: bool):
        # assert(qc.num_qubits > qubit)

        if bit:
            qc.x(qubit)
    
    def _add_one(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        return qc
    
    def _add_on_two_qubits(self, constant_bits):
        assert(len(constant_bits) == 2)
        assert(constant_bits[0] == 1)

        qc = QuantumCircuit(2)
        qc.cx(0,1)
        qc.x(0)
        self._classic_cx(qc, 1, constant_bits[1])

        return qc
    
    def _add_on_three_qubits(self, constant_bits):
        assert(len(constant_bits) == 3)
        assert(constant_bits[0] == 1)

        qc = QuantumCircuit(3)
        self._classic_cx(qc, 0, constant_bits[1])
        self._classic_cx(qc, 1, constant_bits[1])
        qc.ccx(0,1,2)
        self._classic_cx(qc, 0, constant_bits[1])
        qc.cx(0,1)
        qc.x(0)
        a1a2 = constant_bits[1] ^ constant_bits[2]
        self._classic_cx(qc, 1, a1a2)

        return qc


    def m1(self, constant: int, qubit_count: int):
        # no assumptions

        constant = self._constant_to_range(constant, qubit_count)
        
        if constant == 0:
            return QuantumCircuit(qubit_count)
        
        constant, offset = self._remove_zero_lsbs(constant)
        
        adder_circuit = self.m2(constant, qubit_count - offset)

        if offset <= 0:
            return adder_circuit
        
        return self._add_offset(adder_circuit, offset)
        

    def m2(self, constant: int, qubit_count: int):
        assert(constant > 0)
        assert(constant < 2** qubit_count)
        assert(constant % 2 == 1)

        constant_bits = self._int_to_bits(constant, num_bits=qubit_count)
        return self.m3(constant_bits)

    def m3(self, constant_bits):
        qubit_count = len(constant_bits)
        
        if qubit_count == 1:
            return self._add_one()
        
        if qubit_count == 2:
            return self._add_on_two_qubits(constant_bits)
        
        if qubit_count == 3:
            return self._add_on_three_qubits(constant_bits)
        
        return self.m4(constant_bits)
    
    def m4(self, constant_bits: list):
        data_count = len(constant_bits)
        anc_count = data_count - 3

        assert(anc_count > 0)
        assert(constant_bits[0] == 1)

        data = QuantumRegister(data_count, 'data')
        anc  = AncillaRegister(anc_count, 'anc')
        qc   = QuantumCircuit(data, anc, name=f"+{int(''.join(map(str, constant_bits)), 2)}")

        logical_op = LogicalOperator(qc, self.allow_measurement, self.optimize_t_gates)

        self._classic_cx(qc, data[0], constant_bits[1])
        self._classic_cx(qc, data[1], constant_bits[1])

        logical_op.apply_temporary_and(data[0], data[1], anc[0])
        a1a2 = constant_bits[1] ^ constant_bits[2]
        self._classic_cx(qc, anc[0], a1a2)
        self._classic_cx(qc, data[2], constant_bits[2])

        for a in range(1, anc_count):
            logical_op.apply_temporary_and(anc[a-1], data[a+1], anc[a])
            axay = constant_bits[a+1] ^ constant_bits[a+2]
            self._classic_cx(qc, anc[a], axay)
            self._classic_cx(qc, data[a+2], constant_bits[a+2])

        qc.ccx(anc[-1], data[-2], data[-1])
        axay = constant_bits[-2] ^ constant_bits[-1]
        self._classic_cx(qc, data[-1], axay)

        for a in reversed(range(1, anc_count)):
            self._classic_cx(qc, anc[a], constant_bits[a+2])
            qc.cx(anc[a], data[a+2])
            self._classic_cx(qc, anc[a], constant_bits[a+1])
            logical_op.uncompute_temporary_and(anc[a-1], data[a+1], anc[a])

        self._classic_cx(qc, anc[0], constant_bits[2])
        qc.cx(anc[0], data[2])
        self._classic_cx(qc, anc[0], data[1])
        logical_op.uncompute_temporary_and(data[0], data[1], anc[0])

        self._classic_cx(qc, data[0], constant_bits[1])
        qc.cx(data[0], data[1])
        qc.x(data[0])

        return qc
            

