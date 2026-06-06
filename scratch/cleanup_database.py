import re

with open("database.py", "r", encoding="utf-8") as f:
    content = f.read()

# Update cases imports to include CaseComment and CasePresence
old_import = """from db.models.cases import (
    CaseStatus, DocumentType, CaseDeadline, Case, CaseDocument, Attachment, CaseTimeline, CaseNote, AnonymizedShareToken,
)"""

new_import = """from db.models.cases import (
    CaseStatus, DocumentType, CaseDeadline, Case, CaseDocument, Attachment, CaseTimeline, CaseNote, AnonymizedShareToken,
    CaseComment, CasePresence,
)"""

content = content.replace(old_import, new_import)

# Find the start of duplicate definitions: class NotificationStatus
# Find the end of duplicate definitions: class PrecedentMatch representation
start_marker = "class NotificationStatus(str, enum.Enum):"
end_marker = 'return f"<PrecedentMatch(query={self.query_case_id}, precedent={self.precedent_case_id}, type={self.match_type})>"'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    # Find the next newline after end_marker
    actual_end = end_idx + len(end_marker)
    while actual_end < len(content) and content[actual_end] in ('\r', '\n'):
        actual_end += 1
    
    # Slice out the duplicate block
    content = content[:start_idx] + content[actual_end:]
    print("Consolidated database.py successfully!")
else:
    print(f"Failed to find markers: start={start_idx}, end={end_idx}")

with open("database.py", "w", encoding="utf-8") as f:
    f.write(content)
