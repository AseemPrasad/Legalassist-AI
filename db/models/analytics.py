import datetime as dt
import json
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.types import TypeDecorator
from sqlalchemy.orm import relationship
from db.base import Base


class CaseRecord(Base):
    __tablename__ = "case_records"
    id = Column(Integer, primary_key=True)
    hashed_case_id = Column(String(255), unique=True, nullable=False, index=True)
    case_type = Column(String(255), index=True)
    jurisdiction = Column(String(255), index=True)
    court_name = Column(String(255))
    judge_name = Column(String(255))
    plaintiff_type = Column(String(255))
    defendant_type = Column(String(255))
    case_value = Column(String(255))
    outcome = Column(String(255))
    judgment_summary = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc))

    outcome_data = relationship("db.models.analytics.CaseOutcome", back_populates="case_record", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CaseRecord(id={self.id}, hashed_id='{self.hashed_case_id}', outcome='{self.outcome}')>"


class CaseOutcome(Base):
    __tablename__ = "case_outcomes"
    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("case_records.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    appeal_filed = Column(Boolean, default=False, nullable=False)
    appeal_date = Column(DateTime(timezone=True), nullable=True)
    appeal_outcome = Column(String(255), nullable=True)  # appeal_allowed, appeal_rejected, withdrawn, pending
    appeal_success = Column(Boolean, nullable=True)  # True = won, False = lost, None = pending
    time_to_appeal_verdict = Column(Integer, nullable=True)  # days
    appeal_cost = Column(String(255), nullable=True)  # estimated cost range
    additional_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Relationships
    case_record = relationship("db.models.analytics.CaseRecord", back_populates="outcome_data")

    def __repr__(self):
        return f"<CaseOutcome(case_id={self.case_id}, appeal_filed={self.appeal_filed}, appeal_success={self.appeal_success})>"


class CaseAnalytics(Base):
    """Model for aggregated analytics (refreshed periodically)"""
    __tablename__ = "case_analytics"

    id = Column(Integer, primary_key=True)
    case_type = Column(String(255), nullable=False)
    jurisdiction = Column(String(255), nullable=False, index=True)
    court_name = Column(String(255), nullable=True)
    judge_name = Column(String(255), nullable=True)
    
    total_cases = Column(Integer, default=0)
    plaintiff_win_count = Column(Integer, default=0)
    defendant_win_count = Column(Integer, default=0)
    settlement_count = Column(Integer, default=0)
    
    appeals_filed = Column(Integer, default=0)
    appeals_successful = Column(Integer, default=0)
    appeal_success_rate = Column(String(255), default="0%")
    
    avg_case_duration = Column(Integer, nullable=True)
    avg_appeal_duration = Column(Integer, nullable=True)
    avg_appeal_cost = Column(Integer, nullable=True)
    
    last_updated = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    def __repr__(self):
        return f"<CaseAnalytics(jurisdiction={self.jurisdiction}, appeal_success_rate={self.appeal_success_rate})>"


class ModelFeedback(Base):
    """User feedback on model outputs for later training and evaluation"""
    __tablename__ = "model_feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    task = Column(String(100), nullable=False, index=True)
    case_id = Column(Integer, ForeignKey("case_records.id", ondelete="SET NULL"), nullable=True, index=True)
    is_accurate = Column(Boolean, nullable=True, index=True)
    corrected_text = Column(Text, nullable=True)
    feedback_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    case = relationship("db.models.analytics.CaseRecord")

    def __repr__(self):
        return f"<ModelFeedback(model={self.model_name}, task={self.task}, accurate={self.is_accurate})>"


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False, index=True)
    task = Column(String(100), nullable=False, index=True)
    case_type = Column(String(100), nullable=True, index=True)
    jurisdiction = Column(String(100), nullable=True, index=True)
    samples = Column(Integer, default=0)
    accurate_count = Column(Integer, default=0)
    accuracy = Column(String(50), default="0%")
    average_latency_ms = Column(Integer, nullable=True)
    average_cost = Column(Integer, nullable=True)
    last_updated = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    def __repr__(self):
        return f"<ModelPerformance(model={self.model_name}, task={self.task}, accuracy={self.accuracy})>"


class ModelRoutingRule(Base):
    __tablename__ = "model_routing_rule"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    case_type = Column(String(100), nullable=True)
    jurisdiction = Column(String(100), nullable=True)
    min_case_value = Column(String(50), nullable=True)
    task = Column(String(100), nullable=False)
    preferred_model = Column(String(255), nullable=False)
    approved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    def __repr__(self):
        return f"<ModelRoutingRule(name={self.name}, task={self.task}, model={self.preferred_model})>"


class SimilarityFeedback(Base):
    __tablename__ = "similarity_feedback"

    id = Column(Integer, primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    query_signature = Column(String(512), nullable=False, index=True)
    candidate_case_id = Column(Integer, ForeignKey("case_records.id", ondelete="CASCADE"), nullable=False, index=True)
    relevance = Column(Boolean, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    candidate_case = relationship("db.models.analytics.CaseRecord")

    def __repr__(self):
        return f"<SimilarityFeedback(user_id={self.user_id}, candidate_case_id={self.candidate_case_id}, relevance={self.relevance})>"



class SafeVector(TypeDecorator):
    """Dynamically uses pgvector's Vector type on PostgreSQL, falling back to Text on other dialects (SQLite)."""
    impl = Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector
                return dialect.type_descriptor(Vector(1536))
            except ImportError:
                pass
        return dialect.type_descriptor(Text)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        
        # Handle array-like or numpy object conversions
        if hasattr(value, "tolist"):
            value = value.tolist()

        if dialect.name == "postgresql":
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except Exception:
                    pass
            return value
        else:
            if not isinstance(value, str):
                return json.dumps(value)
            return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            if not isinstance(value, str):
                return json.dumps(value)
            return value
        return value


class CaseEmbedding(Base):
    """Model for storing semantic embeddings of cases for similarity search"""
    __tablename__ = "case_embeddings"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    document_id = Column(Integer, ForeignKey("case_documents.id", ondelete="SET NULL"), nullable=True)
    
    # Embedding vector (stored dynamically using SafeVector type decorator)
    embedding_vector = Column(SafeVector, nullable=False)
    embedding_model = Column(String(255), default="text-embedding-3-small")  # Model used to generate
    embedding_dimension = Column(Integer, default=1536)
    
    # Metadata for filtering
    case_type = Column(String(255), nullable=False, index=True)
    jurisdiction = Column(String(255), nullable=False, index=True)
    outcome = Column(String(255), nullable=True, index=True)  # plaintiff_won, defendant_won, settlement, etc.
    
    # Timestamps
    indexed_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Relationships
    case = relationship("db.models.cases.Case")
    document = relationship("db.models.cases.CaseDocument")

    def __repr__(self):
        return f"<CaseEmbedding(case_id={self.case_id}, model={self.embedding_model})>"


class CaseIssue(Base):
    """Model for tracking legal issues/topics extracted from cases"""
    __tablename__ = "case_issues"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Issue details
    issue_name = Column(String(255), nullable=False, index=True)  # e.g., "wrongful termination", "property dispute"
    issue_description = Column(Text, nullable=True)
    issue_category = Column(String(255), nullable=True, index=True)  # civil, criminal, family, labor, etc.
    
    # Confidence score (0-1) from extraction model
    confidence_score = Column(String(50), default="1.0")  # JSON-safe string
    
    # Metadata
    extracted_from_document = Column(Integer, ForeignKey("case_documents.id", ondelete="SET NULL"), nullable=True)
    extraction_method = Column(String(255), default="llm")  # llm, keyword, manual
    
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Relationships
    case = relationship("db.models.cases.Case")
    document = relationship("db.models.cases.CaseDocument")
    arguments = relationship("db.models.analytics.CaseArgument", back_populates="issue", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("case_id", "issue_name", name="uq_case_issue"),)

    def __repr__(self):
        return f"<CaseIssue(case_id={self.case_id}, issue={self.issue_name})>"


class CaseArgument(Base):
    """Model for tracking legal arguments used in cases"""
    __tablename__ = "case_arguments"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    issue_id = Column(Integer, ForeignKey("case_issues.id", ondelete="CASCADE"), nullable=True)
    
    # Argument details
    argument_text = Column(Text, nullable=False)  # The actual argument made
    argument_type = Column(String(255), nullable=True, index=True)  # witness_testimony, precedent_citation, legal_principle, etc.
    
    # Whether the argument succeeded in this case
    argument_succeeded = Column(Boolean, nullable=True)  # True=won, False=lost, None=unknown
    
    # Supporting evidence
    supporting_evidence = Column(Text, nullable=True)  # Quote or reference from judgment
    citation_references = Column(JSON, nullable=True)  # List of law citations
    
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)

    # Relationships
    case = relationship("db.models.cases.Case")
    issue = relationship("db.models.analytics.CaseIssue", back_populates="arguments")

    def __repr__(self):
        return f"<CaseArgument(case_id={self.case_id}, type={self.argument_type})>"


class KnowledgeGraphEdge(Base):
    """Model for building a knowledge graph: Case → Issue → Argument → Outcome"""
    __tablename__ = "knowledge_graph_edges"

    id = Column(Integer, primary_key=True)
    
    # Source: Issue
    issue_id = Column(Integer, ForeignKey("case_issues.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Edge: Argument
    argument_id = Column(Integer, ForeignKey("case_arguments.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Target: Outcome
    case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    outcome = Column(String(255), nullable=False, index=True)  # plaintiff_won, defendant_won, settlement, etc.
    
    # Weight: How strongly the argument led to this outcome (frequency + confidence)
    weight = Column(String(50), default="1.0")  # String for JSON safety
    
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    # Relationships
    issue = relationship("db.models.analytics.CaseIssue")
    argument = relationship("db.models.analytics.CaseArgument")
    case = relationship("db.models.cases.Case")

    __table_args__ = (UniqueConstraint("issue_id", "argument_id", "case_id", name="uq_graph_edge"),)

    def __repr__(self):
        return f"<KnowledgeGraphEdge(issue={self.issue_id}, argument={self.argument_id}, outcome={self.outcome})>"


class PrecedentMatch(Base):
    """Model for storing precedent matching results for quick lookup"""
    __tablename__ = "precedent_matches"

    id = Column(Integer, primary_key=True)
    
    # Case that's being analyzed
    query_case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Similar precedent case
    precedent_case_id = Column(Integer, ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Matching type
    match_type = Column(String(255), nullable=False, index=True)  # similar_case, precedent_with_winning_argument, etc.
    
    # Similarity score (0-1)
    similarity_score = Column(String(50), default="0.0")  # String for JSON safety
    
    # Reason for match
    match_reason = Column(Text, nullable=True)  # "Similar issues", "Winning argument", etc.
    
    # Metadata about the match
    shared_issues = Column(JSON, nullable=True)  # List of shared issue names
    shared_arguments = Column(JSON, nullable=True)  # List of matching argument texts
    precedent_outcome = Column(String(255), nullable=True)  # Outcome in precedent case
    
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Cache expiration

    # Relationships
    query_case = relationship("db.models.cases.Case", foreign_keys=[query_case_id])
    precedent_case = relationship("db.models.cases.Case", foreign_keys=[precedent_case_id])

    __table_args__ = (UniqueConstraint("query_case_id", "precedent_case_id", "match_type", name="uq_precedent_match"),)

    def __repr__(self):
        return f"<PrecedentMatch(query={self.query_case_id}, precedent={self.precedent_case_id}, type={self.match_type})>"


class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    id = Column(Integer, primary_key=True)
    jti = Column(String(255), unique=True, nullable=False, index=True)
    revoked_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)

    def __repr__(self):
        return f"<RevokedToken(jti={self.jti})>"
