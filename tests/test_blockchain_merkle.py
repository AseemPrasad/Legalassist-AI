from core.blockchain_sim import BlockchainSimulator


def test_merkle_tree_empty():
    sim = BlockchainSimulator()
    # Sim has genesis block automatically
    root = sim.get_merkle_root()
    assert len(root) == 64


def test_merkle_tree_multiple_leaves():
    sim = BlockchainSimulator()
    sim.append_data_hash("hash1")
    sim.append_data_hash("hash2")
    sim.append_data_hash("hash3")
    sim.append_data_hash("hash4")
    
    root = sim.get_merkle_root()
    assert len(root) == 64
    
    # Generate and verify proof for hash2
    proof = sim.get_merkle_proof("hash2")
    assert len(proof) > 0
    
    verified = BlockchainSimulator.verify_merkle_proof("hash2", proof, root)
    assert verified is True


def test_merkle_tree_tamper():
    sim = BlockchainSimulator()
    sim.append_data_hash("hash1")
    sim.append_data_hash("hash2")
    
    root = sim.get_merkle_root()
    proof = sim.get_merkle_proof("hash2")
    
    # Tampering target or sibling should fail verification
    assert BlockchainSimulator.verify_merkle_proof("hash_tampered", proof, root) is False
