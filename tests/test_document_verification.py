import base64
from core.blockchain_sim import BlockchainSimulator
from core.document_verifier import register_document, verify_document, compute_document_hash


def test_register_and_verify_document():
    chain = BlockchainSimulator()
    data = b"Test PDF content for hashing"
    reg = register_document(data, chain=chain, metadata={"filename": "test.pdf"})
    assert "data_hash" in reg
    dh = reg["data_hash"]
    assert dh == compute_document_hash(data)

    ver = verify_document(data, chain=chain)
    assert ver["found"] is True
    assert ver["data_hash"] == dh
    assert isinstance(ver.get("proof"), list)


def test_verify_missing_document():
    chain = BlockchainSimulator()
    data = b"Unregistered content"
    ver = verify_document(data, chain=chain)
    assert ver["found"] is False
    assert "data_hash" in ver
