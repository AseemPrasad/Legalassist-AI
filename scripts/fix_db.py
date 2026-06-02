import re

with open('database.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove index=True from primary_key=True
content = re.sub(r'Column\(Integer,\s*primary_key=True,\s*index=True\)', r'Column(Integer, primary_key=True)', content)

# 2. Add timezone=True to DateTime
content = re.sub(r'Column\(DateTime,', r'Column(DateTime(timezone=True),', content)

# 3. Add lengths to String columns
content = re.sub(r'Column\(String([,\)])', r'Column(String(255)\1', content)

# 4. Add ondelete="CASCADE" to ForeignKeys
content = re.sub(r'ForeignKey\("([^"]+)"\)', r'ForeignKey("\1", ondelete="CASCADE")', content)

with open('database.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done!")
