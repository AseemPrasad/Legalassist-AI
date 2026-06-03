import ast
import os

def merge():
    # Read database_resolved.py as the clean baseline shim
    with open('database_resolved.py', encoding='utf-8') as f:
        resolved_content = f.read()

    # Read old database.py (backup copy or before merge)
    import subprocess
    old_content = subprocess.check_output(['git', 'show', 'HEAD:database.py'], text=True)

    # Parse both files using AST
    tree_old = ast.parse(old_content)
    tree_resolved = ast.parse(resolved_content)

    # Find functions and classes in database.py
    old_nodes = {}
    for node in tree_old.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            old_nodes[node.name] = node

    unique_names = [
        'IdempotencyKeyStatus',
        'IdempotencyKey',
        'CaseComment',
        'CasePresence',
        'reserve_idempotency_key',
        'set_idempotency_response',
        'get_idempotency_response',
        'reserve_notification',
        'update_notification_result',
        'aggregate_model_performance',
        'schedule_token_cleanup',
        'submit_similarity_feedback',
        'get_similarity_feedback',
        'create_case_comment',
        'get_case_comments',
        'upsert_case_presence',
        'get_case_presence'
    ]

    old_lines = old_content.splitlines()
    appended_code = []

    # Extra imports needed by these functions/classes
    extra_imports = """
import enum
from sqlalchemy import UniqueConstraint, ForeignKey, Column, Integer, String, DateTime, Text, JSON, Enum as SQLEnum, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.exc import IntegrityError
from db.models import (
    ModelPerformance,
    RevokedToken,
    SimilarityFeedback
)
"""
    appended_code.append(extra_imports)

    for name in unique_names:
        if name in old_nodes:
            node = old_nodes[name]
            # AST line numbers are 1-based, slicing is 0-based
            code = "\n".join(old_lines[node.lineno - 1 : node.end_lineno])
            appended_code.append(code)

    # Combine resolved content and unique function/class implementations
    relationship_injection = """
# Dynamic relationships injection to support legacy collaborative features on Case model
from db.models.cases import Case
from sqlalchemy.orm import relationship

Case.comments = relationship("CaseComment", back_populates="case", cascade="all, delete-orphan", order_by="CaseComment.created_at")
Case.presence_updates = relationship("CasePresence", back_populates="case", cascade="all, delete-orphan")
"""
    merged_content = resolved_content + "\n\n" + relationship_injection + "\n\n" + "\n\n".join(appended_code) + "\n"

    # Write back to database.py
    with open('database.py', 'w', encoding='utf-8') as f:
        f.write(merged_content)
    print("Merge completed successfully!")

if __name__ == '__main__':
    merge()
