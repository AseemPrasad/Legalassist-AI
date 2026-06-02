#!/usr/bin/env python3
"""
Local Environment Validation Script

Helper script to quickly verify environment variables, database connections,
and key dependencies needed to run LegalAssist AI locally.
"""
import os
import sys

def validate():
    print("=== LegalAssist AI Environment Validation ===")
    
    # 1. Check virtualenv
    is_venv = hasattr(sys, 'real_prefix') or (sys.base_prefix != sys.prefix)
    print(f"[*] Virtual environment active: {is_venv}")
    
    # 2. Check essential files
    env_exists = os.path.exists(".env")
    print(f"[*] .env file exists: {env_exists}")
    
    # 3. Check critical configurations
    # We try loading configurations securely
    try:
        from api.config import Config
        print(f"[*] App Environment: {Config.APP_ENV}")
        print(f"[*] Database URL configured: {bool(Config.DATABASE_URL)}")
        print(f"[*] Default Model: {Config.DEFAULT_MODEL}")
    except Exception as e:
        print(f"[!] Error loading application config: {e}")
        return False
        
    print("[*] Validation completed successfully.")
    return True

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
