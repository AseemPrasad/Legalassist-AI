import sys
import os
from datetime import datetime

# Add the project root to sys.path
sys.path.append(os.getcwd())

from api.auth import create_api_key_record, verify_api_key

def test_salted_hashing():
    print("Testing salted hashing for API keys...")
    
    # Create a new API key record
    key, record = create_api_key_record("Test Key")
    
    print(f"Generated Key: {key}")
    print(f"Key Hash: {record.key_hash}")
    print(f"Key Salt: {record.key_salt}")
    
    # Verify the key
    is_valid = verify_api_key(key, record.key_salt, record.key_hash)
    print(f"Verification with correct key: {is_valid}")
    assert is_valid == True
    
    # Verify with incorrect key
    is_valid_wrong = verify_api_key("wrong-key", record.key_salt, record.key_hash)
    print(f"Verification with wrong key: {is_valid_wrong}")
    assert is_valid_wrong == False
    
    # Ensure hash is not the same as unsalted hash (SHA-256)
    import hashlib
    unsalted_hash = hashlib.sha256(key.encode()).hexdigest()
    print(f"Unsalted Hash: {unsalted_hash}")
    assert record.key_hash != unsalted_hash
    print("Salted hashing verified successfully!")

if __name__ == "__main__":
    try:
        test_salted_hashing()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
