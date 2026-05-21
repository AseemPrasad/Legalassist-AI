import os
path = r'c:\Users\Rushabh Mahajan\Documents\GitHub\Legalassist-AI\.streamlit\secrets.toml'
if os.path.exists(path):
    os.remove(path)

content = """# LegalAssist AI - Streamlit Secrets Configuration

OPENROUTER_API_KEY = "test_key"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"

TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_FROM_NUMBER = ""

SENDGRID_API_KEY = ""
SENDGRID_FROM_EMAIL = ""

JWT_SECRET = ""
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24
"""

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('File recreated successfully')
