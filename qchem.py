import netket as nk
import pubchempy as pcp
from pyscf import gto, scf, fci, cc, mcscf, ci
import netket.experimental as nkx
from netket.operator import FermionOperator2ndJax
from netket.experimental.operator import ParticleNumberAndSpinConservingFermioperator2nd, ParticleNumberConservingFermioperator2nd
import jax.numpy as jnp
import networkx as nx
from .pubchem_cached import load_cached_compound

class PCMolecule:
    """
    Initialize a model from a PubChem CID
    Defaults
    """

    def __init__(self, mol, mo_coeff, conserve_spin=True,):
        """
        Initialize a PCMolecule.

        Args:
            mol: PySCF molecule object
            mo_coeff: Molecular orbital coefficients
            conserve_spin: If True, use ParticleNumberAndSpinConservingFermioperator2nd.
                          If False, use ParticleNumberConservingFermioperator2nd.
                          Default is True.
        """
        #build up the second-quantized molecular Hamiltonian

        # ha_pyscf = nkx.operator.from_pyscf_molecule(
        #     mol, implementation=FermionOperator2ndJax, mo_coeff=mo_coeff,
        # ).to_jax_operator()
        if conserve_spin:
            # self.hamiltonian = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(
            #     ha_pyscf
            # )
            self.hamiltonian = ParticleNumberAndSpinConservingFermioperator2nd.from_pyscf_molecule(mol, mo_coeff=mo_coeff,)
        else:
            # self.hamiltonian = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(
            #     ha_pyscf
            # )
            self.hamiltonian = ParticleNumberConservingFermioperator2nd.from_pyscf_molecule(mol, mo_coeff=mo_coeff,)
        self.hilbert_space = self.hamiltonian.hilbert

        # make sampling graph
        _, Tij, Vijkl = nkx.operator.pyscf.TV_from_pyscf_molecule(
            mol, mo_coeff=mo_coeff
        )
        No = Tij.shape[0] // 2
        Tij = Tij.todense()
        Tij_up = Tij[:No, :No]
        Tij_down = Tij[No:, No:]
        T_tot = jnp.abs(Tij_up) + jnp.abs(Tij_down)

        Vijkl = Vijkl.todense()
        Vt = jnp.zeros((2 * No, 2 * No))
        for i in range(2 * No):
            Vt = Vt + jnp.abs(Vijkl[i, :, i, :])
        Vt_od = jnp.abs(Vt - jnp.diag(jnp.diagonal(Vt)))
        Vt2 = Vt_od[:No, :No] + Vt_od[No:, No:]

        # Compute the combined adjacency matrix based on the condition
        combined_adjacency = (Vt2 + T_tot) > 1 # 0
        combined_adjacency = combined_adjacency.at[
            jnp.diag_indices_from(combined_adjacency)
        ].set(0)
        adj_matrix = jnp.array(combined_adjacency).astype(int)

        # Create a NetworkX graph from the adjacency matrix
        G = nx.from_numpy_array(adj_matrix)

        # connect medians of each cluster to each other
        components = list(nx.connected_components(G))
        medians = []
        for comp in components:
            sorted_nodes = sorted(comp)
            median_idx = len(sorted_nodes) // 2
            median_node = sorted_nodes[median_idx]
            medians.append(median_node)
        medians.sort()
        for i in range(len(medians) - 1):
            G.add_edge(medians[i], medians[i + 1])

        g = nk.graph.Graph.from_networkx(G)
        # g = nk.graph.Chain(self.hilbert_space.n_orbitals, pbc=False)
        self.graph = nk.graph.disjoint_union(
            g, g
        )  # only relevant for the fermihop sampler

    
    @staticmethod
    def molecule(cid, basis=None):
        # visit the PuChem library to understand the molecule's geometry

        try:
            compound = pcp.get_compounds(cid, "cid", record_type="3d")[0]
            geom = "3d"
        except:
            print("using 2d")
            compound = pcp.get_compounds(cid, "cid", record_type="2d")[0]
            geom = "2d"

        
        mol, mo_coeff, mf = build_molecule_geometry(compound, geom, basis)

        return mol, mo_coeff, mf

    @staticmethod
    def molecule_cached(cid, dir, basis=None):
        # visit the PuChem library to understand the molecule's geometry
        compound = load_cached_compound(cid, dir)
        geom = compound.geom

        
        mol, mo_coeff, mf = build_molecule_geometry(compound, geom, basis)

        return mol, mo_coeff, mf

def build_molecule_geometry(compound, geom, basis=None):
    # 2. Extract atomic coordinates
    geometry = []

    for atom in compound.atoms:
        symbol = atom.element
        if geom == "3d":
            x, y, z = atom.x, atom.y, atom.z
        elif geom == "2d":
            x, y, z = atom.x, atom.y, 0.0
        geometry.append(f"{symbol} {x} {y} {z}")

    # Convert to PySCF format
    mol_geometry = "\n".join(geometry)

    # 3. Define the molecule in PySCF
    mol = gto.Mole()
    mol.atom = mol_geometry
    mol.basis = basis   #"STO-3G"  # Choose a reasonable basis set
    mol.unit = "angstrom"  # Coordinates are in Ångströms
    mol.spin = 0 
    mol.charge = 0
    mol.build()

    # 4. Run Hartree-Fock calculation
    mf = scf.HF(mol).run(verbose=0)
    mo_coeff = mf.mo_coeff
    mf.kernel()
    print(f"Hartree-Fock energy: {mf.e_tot}")

    # 5. Compute Coupled Cluster Single-Double excitation (CCSD) energy
    ccsd = cc.ccsd.CCSD(mf).run()
    print(f"CCSD energy: {ccsd.e_tot}")

    # 6. Compute Full Configuration Interaction (FCI) energy
    # cisolver = fci.FCI(mol, mf.mo_coeff)
    # E_fci = cisolver.kernel()[0]
    # print(f"FCI energy: {E_fci}")
    return mol, mo_coeff, mf
