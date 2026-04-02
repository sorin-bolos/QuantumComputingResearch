import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate

class Mps:
    def __init__(self, function_1d):
        self.function_1d = function_1d

    def get_discretized_function(self, n_qubits, max_range):
        num_points = 2 ** n_qubits
        step_size = max_range / num_points
        points = np.arange(num_points) * step_size + step_size/2  # cell centers, avoiding x=0

        return [self.function_1d(point) for point in points]
    
    def get_normalized_amplitudes(self, n_qubits, max_range):
        func_values = self.get_discretized_function(n_qubits, max_range)
        norm = np.linalg.norm(func_values)
        if norm < 1e-30:
            return np.zeros_like(func_values)
        return func_values / norm
    
    def compute_max_bond_dimension(self, n_qubits, max_range, threshold=1e-12):
        """
        Compute the maximum MPS bond dimension across all bipartitions
        of an amplitude-encoded 1D function on n_qubits.
        
        The state vector |psi> = sum_j f(x_j)|j> (normalized) is reshaped
        into a tensor with n_qubits indices of dimension 2 each.
        We compute the SVD across every bipartition and find the max
        number of singular values above threshold.
        """
        psi = self.get_normalized_amplitudes(n_qubits, max_range)
        
        max_chi = 1
        for cut in range(1, n_qubits):
            left_dim = 2**cut
            right_dim = 2**(n_qubits - cut)
            mat = psi.reshape(left_dim, right_dim)
            
            sv = np.linalg.svd(mat, compute_uv=False)
            
            chi = np.sum(sv > threshold)
            max_chi = max(max_chi, chi)
        
        return max_chi
    
    def verify_circuit(self, qc, n_qubits, target_state):
        """Verify circuit produces the correct state on physical qubits."""
        sv = Statevector.from_instruction(qc)
        sv_array = np.array(sv)
    
        # Bond qubits = 0 → indices 0 to 2^n_qubits - 1
        psi_circuit = sv_array[:2**n_qubits]
        leakage = np.linalg.norm(sv_array[2**n_qubits:])
    
        psi_target = target_state / np.linalg.norm(target_state)
        norm_circ = np.linalg.norm(psi_circuit)
        psi_circ_n = psi_circuit / norm_circ if norm_circ > 1e-10 else psi_circuit
    
        fidelity = np.abs(np.vdot(psi_target, psi_circ_n))**2
        return fidelity, leakage, psi_circ_n
    
    def generate_mps_circuit(self, n_qubits, max_range, label='mps'):
        normalized_f = self.get_normalized_amplitudes(n_qubits, max_range)
        tensors = self._decompose_to_mps_right_canonical(n_qubits, normalized_f)
        qc = self._build_mps_circuit(tensors, n_qubits, label)
        return qc
    
    def _decompose_to_mps_right_canonical(self, n_qubits, normalized_f, threshold=1e-12) -> list:
        """
        Right-canonical MPS via SVD from right to left.
        
        numpy C-order reshape convention:
        psi[j] where j = sigma_0 * 2^{n-1} + sigma_1 * 2^{n-2} + ... + sigma_{n-1}
        sigma_0 = MSB (slowest varying), sigma_{n-1} = LSB (fastest varying)
        
        We peel off from the right (LSB first): site n-1, then n-2, etc.
        
        Result: tensors[i] has shape (chi_i, 2, chi_{i+1}).
        - tensors[1..n-1] are right-canonical: sum_sigma A^sigma (A^sigma)† = I
        - tensors[0] carries the normalization (singular values absorbed)
        
        MPS site i corresponds to sigma_i (bit position n-1-i from the right,
        or Qiskit qubit n-1-i).
        """
        tensors = [None] * n_qubits
        remainder = normalized_f.copy()
        chi_right = 1
    
        for site in range(n_qubits - 1, 0, -1):
            # remainder has shape (2^{site+1}, chi_right) conceptually
            # but stored flat as length 2^{site+1} * chi_right
            # (for site = n-1, chi_right = 1, so it's just psi)
            
            # Reshape to (2^site, 2 * chi_right) to split off sigma_{site}
            n_left = 2**site
            mat = remainder.reshape(n_left, 2 * chi_right)
    
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            chi_left = int(np.sum(S > threshold))
            U = U[:, :chi_left]
            S = S[:chi_left]
            Vh = Vh[:chi_left, :]
    
            # Vh has shape (chi_left, 2 * chi_right) with orthonormal rows
            # Reshape to (chi_left, 2, chi_right)
            tensors[site] = Vh.reshape(chi_left, 2, chi_right)
    
            # Pass singular values to the left
            remainder = U @ np.diag(S)  # shape (2^site, chi_left)
            remainder = remainder.flatten()
            chi_right = chi_left
    
        # First tensor: remainder has shape (2 * chi_right,)
        # = (1, 2, chi_1) — carries the normalization
        tensors[0] = remainder.reshape(1, 2, chi_right)
    
        return tensors
    
    def _contract_mps_right_canonical(self, tensors) -> np.ndarray:
        """
        Contract MPS tensors to reconstruct the state vector.
        
        Processes left to right. Output index:
        j = sigma_0 * 2^{n-1} + sigma_1 * 2^{n-2} + ... + sigma_{n-1}
        """
        state = tensors[0][0, :, :]  # shape (2, chi_1)
    
        for i in range(1, len(tensors)):
            chi_next = tensors[i].shape[2]
            d = state.shape[0]
            new_state = np.zeros((d * 2, chi_next), dtype=complex)
            for sigma in range(2):
                # new_state[old_idx * 2 + sigma] = state[old_idx] @ A[i][:, sigma, :]
                new_state[np.arange(d) * 2 + sigma, :] = state @ tensors[i][:, sigma, :]
            state = new_state
    
        return state.flatten()
    
    def _mps_tensor_to_unitary(self, tensor, n_bond_qubits) -> np.ndarray:
        """
        Convert one right-canonical MPS tensor to a unitary gate.
        
        tensor: shape (chi_left, 2, chi_right)
        
        Gate acts on (1 physical + n_bond) qubits.
        Maps: |0>_phys |alpha_in>_bond → sum A[alpha_in, sigma, alpha_out] |sigma>_phys |alpha_out>_bond
        
        Flat index: alpha * 2 + sigma (physical qubit is LSB, Qiskit convention).
        
        Right-canonical property guarantees the isometry columns are orthonormal.
        """
        chi_left, _, chi_right = tensor.shape
        gate_dim = 2**(n_bond_qubits + 1)
    
        # Build isometry columns (one per input alpha_left)
        iso_columns = []
        for a_left in range(chi_left):
            col = np.zeros(gate_dim, dtype=complex)
            for sigma in range(2):
                for a_right in range(chi_right):
                    col[a_right * 2 + sigma] = tensor[a_left, sigma, a_right]
            iso_columns.append(col)
    
        # Verify orthonormality (should hold for right-canonical tensors)
        if len(iso_columns) > 1:
            gram = np.zeros((len(iso_columns), len(iso_columns)), dtype=complex)
            for i, ci in enumerate(iso_columns):
                for j, cj in enumerate(iso_columns):
                    gram[i, j] = np.vdot(ci, cj)
            ortho_err = np.max(np.abs(gram - np.eye(len(iso_columns))))
            if ortho_err > 1e-8:
                print(f"  WARNING: ortho error = {ortho_err:.2e} "
                    f"(chi_l={chi_left}, chi_r={chi_right})")
    
        # Complete to unitary via QR on [iso_columns | random vectors]
        # This is more numerically stable than sequential Gram-Schmidt
        
        # Build matrix with iso columns at correct positions, random elsewhere
        M = np.random.randn(gate_dim, gate_dim) + 0j
        for j in range(chi_left):
            M[:, 2 * j] = iso_columns[j]
    
        # QR factorization: M = Q R
        # Q is unitary with Q's columns forming an orthonormal basis
        # But Q[:, 2*j] won't exactly equal iso_columns[j] because QR
        # modifies all columns. Instead, use a two-step approach:
    
        # Step 1: Build projector onto complement of isometry space
        V_iso = np.column_stack(iso_columns)  # (gate_dim, chi_left)
        P_perp = np.eye(gate_dim) - V_iso @ V_iso.conj().T  # projector onto complement
    
        # Step 2: Project random vectors and orthonormalize to get complement basis
        n_complement = gate_dim - chi_left
        R_rand = np.random.randn(gate_dim, n_complement)
        R_proj = P_perp @ R_rand
        Q_comp, _ = np.linalg.qr(R_proj)
        # Q_comp has shape (gate_dim, n_complement) — first n_complement cols are orthonormal
        complement_vecs = [Q_comp[:, k] for k in range(n_complement)]
    
        # Step 3: Assemble unitary
        U = np.zeros((gate_dim, gate_dim), dtype=complex)
        placed = set()
        for j in range(chi_left):
            pos = 2 * j
            U[:, pos] = iso_columns[j]
            placed.add(pos)
    
        remaining = [p for p in range(gate_dim) if p not in placed]
        for pos, vec in zip(remaining, complement_vecs):
            U[:, pos] = vec
    
        # Verify unitarity
        err = np.max(np.abs(U @ U.conj().T - np.eye(gate_dim)))
        if err > 1e-8:
            raise ValueError(f"Unitary error = {err:.2e}")
    
        return U
 
    
    def _build_mps_circuit(self, tensors, n_qubits, label='mps') -> QuantumCircuit:
        """
        Build circuit from MPS tensors.
        
        Qubit layout: [q_0, ..., q_{n-1}, b_0, ..., b_{m-1}]
        MPS site i → Qiskit qubit (n-1-i):
        site 0 (sigma_0, MSB) → qubit n-1
        site n-1 (sigma_{n-1}, LSB) → qubit 0
        """
        bond_dims = [1] + [t.shape[2] for t in tensors]
        max_chi = max(bond_dims)
        n_bond = max(int(np.ceil(np.log2(max(max_chi, 2)))), 1)
    
        total_qubits = n_qubits + n_bond
        qc = QuantumCircuit(total_qubits, name='mps')
    
        phys_qubits = list(range(n_qubits))
        bond_qubits = list(range(n_qubits, total_qubits))
    
        for i, tensor in enumerate(tensors):
            U = self._mps_tensor_to_unitary(tensor, n_bond)
            gate = UnitaryGate(U, label=f'{label}_{i}')
            target_phys = phys_qubits[n_qubits - 1 - i]
            target = [target_phys] + bond_qubits
            qc.append(gate, target)
    
        return qc
