from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister

class LogicalOperator:
    """Logical operators for quantum circuits."""
    def __init__(self, circuit: QuantumCircuit, allow_measurement: bool = True, optimize_t_gates: bool =True):
        self.circuit = circuit
        self.allow_measurement = allow_measurement
        self.optimize_t_gates = optimize_t_gates

    def apply_temporary_and(self, operand1: int, operand2: int, target: int):
        if (not self.optimize_t_gates):
            self.circuit.ccx(operand1, operand2, target)
            return

        and_circuit = self._temporary_and()
        qubit_map = [operand1, operand2, target]
        self.circuit.compose(and_circuit, qubits=qubit_map, inplace=True)

    def uncompute_temporary_and(self, operand1: int, operand2: int, target: int):
        if (not self.optimize_t_gates):
            self.circuit.ccx(operand1, operand2, target)
            return

        if self.allow_measurement:
            uncompute_circuit = self._temporary_and_uncompute_with_measurement()
            qubit_map = [operand1, operand2, target]
            self.circuit.compose(uncompute_circuit, qubits=qubit_map, inplace=True)
            return

        and_dagger_circuit = self._temporary_and_dagger()
        qubit_map = [operand1, operand2, target]
        self.circuit.compose(and_dagger_circuit, qubits=qubit_map, inplace=True)

    @staticmethod
    def _temporary_and() -> QuantumCircuit:
        """
        Ref: 	arXiv:1709.06648
        Implements a logical AND with the restriction that the target qubit 
        is initialized in the |0> state 
        and is returned to the |0> state at the end of the operation.
        """   
        qc = QuantumCircuit(3)
        operand1 = 0
        operand2 = 1
        target = 2

        qc.h(target)
        qc.t(target) 
        qc.cx(operand1, target) 
        qc.cx(operand2, target)
        qc.cx(target, operand1)
        qc.cx(target, operand2)
        qc.tdg(operand1) 
        qc.tdg(operand2)
        qc.t(target)
        qc.cx(target, operand1)
        qc.cx(target, operand2)
        qc.h(target)
        qc.s(target)
        
        return qc

    @staticmethod
    def _temporary_and_dagger() -> QuantumCircuit:
        qc = QuantumCircuit(3)
        operand1 = 0
        operand2 = 1
        target = 2

        qc.sdg(target)
        qc.h(target)
        qc.cx(target, operand1)
        qc.cx(target, operand2)
        qc.t(operand1) 
        qc.t(operand2)
        qc.tdg(target)
        qc.cx(target, operand1)
        qc.cx(target, operand2) 
        qc.cx(operand2, target)
        qc.cx(operand1, target)
        qc.tdg(target)
        qc.h(target)
        
        return qc
    
    @staticmethod
    def _temporary_and_uncompute_with_measurement():
        data = QuantumRegister(3)
        mid   = ClassicalRegister(1, 'mid')
        qc    = QuantumCircuit(data, mid, name="uncompute and")
        clbit = mid[0]
        operand1 = 0
        operand2 = 1
        target = 2

        qc.h(target)
        qc.measure(target, clbit)
        with qc.if_test((clbit, 1)):
            qc.cz(operand1, operand2)

        return qc