"""
Pydantic models for API requests/responses
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr


# ============================================================================
# Authentication Models
# ============================================================================

class TokenRequest(BaseModel):
    """OAuth2 token request"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """OAuth2 token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyCreate(BaseModel):
    """Create API key"""
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response"""
    id: str
    name: str
    key: str  # Only shown on creation
    created_at: datetime
    expires_at: Optional[datetime]


# ============================================================================
# Document Analysis Models
# ============================================================================

class DocumentAnalysisRequest(BaseModel):
    """Request to analyze a document"""
    file_url: Optional[str] = None
    file_path: Optional[str] = None
    text: Optional[str] = None
    document_type: str = Field("unknown", description="Type of document (contract, lawsuit, etc.)")
    extract_remedies: bool = True
    extract_deadlines: bool = True
    extract_obligations: bool = True
    language: str = "en"


class RemediaryItem(BaseModel):
    """A remedy or legal action"""
    type: str
    description: str
    applicable_date: Optional[str]
    jurisdiction: str
    priority: str = "medium"


class DeadlineItem(BaseModel):
    """An important deadline"""
    title: str
    description: str
    date: datetime
    reminder_days: int = 7
    jurisdiction: str


class DocumentAnalysisSummary(BaseModel):
    """Summary of document analysis"""
    document_id: str
    title: str
    document_type: str
    summary: str
    key_points: List[str]
    remedies: List[RemediaryItem]
    deadlines: List[DeadlineItem]
    obligations: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0, description="Model confidence for extraction quality (0.0-1.0)")
    remedies_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence for remedies extraction (0.0-1.0)")
    remedies_evidence_spans: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_time_seconds: float


class AnalysisJobResponse(BaseModel):
    """Response for async analysis job"""
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    result_url: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Case Search Models
# ============================================================================

class CaseSearchRequest(BaseModel):
    """Search for similar cases"""
    case_number: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    jurisdiction: str = "US"
    case_type: str = "general"
    court_name: Optional[str] = None
    judge_name: Optional[str] = None
    plaintiff_type: Optional[str] = None
    defendant_type: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    relevance_threshold: float = Field(0.7, ge=0, le=1)
    query_signature: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)


class CaseResult(BaseModel):
    """A case search result"""
    case_id: str
    case_number: str
    title: str
    year: int
    jurisdiction: str
    case_type: str
    summary: str
    verdict: str
    relevance_score: float = Field(ge=0, le=1)
    appeal_success_rate: Optional[float] = Field(None, ge=0, le=1)
    url: Optional[str] = None


class CaseSearchResponse(BaseModel):
    """Search results"""
    total_results: int
    results: List[CaseResult]
    search_time_seconds: float
    appeal_success_rate: Optional[float] = Field(None, ge=0, le=1)
    appealed_cases: int = 0
    appeal_successful_cases: int = 0


class SimilarityFeedbackRequest(BaseModel):
    """Feedback payload for a similarity result"""
    candidate_case_id: int = Field(..., ge=1)
    query_signature: Optional[str] = None
    relevance: bool


class SimilarityFeedbackResponse(BaseModel):
    """Similarity feedback persistence response"""
    success: bool
    saved_at: datetime
    feedback_id: int


class ModelFeedbackRequest(BaseModel):
    model_name: str
    task: str
    case_id: Optional[int] = None
    is_accurate: Optional[bool] = None
    corrected_text: Optional[str] = None
    feedback_notes: Optional[str] = None


class ModelFeedbackResponse(BaseModel):
    success: bool
    feedback_id: int
    saved_at: datetime


class ModelPerformanceItem(BaseModel):
    model_name: str
    task: str
    case_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    samples: int
    accurate_count: int
    accuracy: str


class ModelPerformanceResponse(BaseModel):
    items: List[ModelPerformanceItem]


# ============================================================================
# Case Timeline Models
# ============================================================================

class CaseEvent(BaseModel):
    """An event in case timeline"""
    date: datetime
    event_type: str  # filing, hearing, decision, appeal, etc.
    description: str
    court: Optional[str] = None
    judge: Optional[str] = None
    location: Optional[str] = None
    documents: List[str] = Field(default_factory=list)


class CaseTimeline(BaseModel):
    """Case history and timeline"""
    case_id: str
    case_number: str
    title: str
    status: str  # open, closed, appealed, etc.
    created_at: datetime
    updated_at: datetime
    events: List[CaseEvent]
    total_events: int
    duration_years: float


class CaseNoteDraftRequest(BaseModel):
    case_id: Optional[str] = None
    note_text: str = Field(..., min_length=1)


class CaseNotePublishRequest(BaseModel):
    case_id: Optional[str] = None
    note_text: Optional[str] = None


class CaseNoteVersionItem(BaseModel):
    version_number: int
    note_text: str
    change_type: str
    changed_by_user_id: str
    changed_by_email: Optional[str] = None
    created_at: datetime
    version_metadata: Optional[Dict[str, Any]] = None


class CaseNoteHistoryResponse(BaseModel):
    case_id: str
    case_number: str
    title: str
    total_versions: int
    versions: List[CaseNoteVersionItem]


class AnonymizedShareCreateRequest(BaseModel):
    scope: str = Field("personal_identifiers", min_length=1, max_length=255)
    expires_in_hours: int = Field(72, ge=1, le=8760)


class AnonymizedShareResponse(BaseModel):
    token: str
    anonymized_id: str
    scope: str
    share_url: str
    expires_at: datetime


# ============================================================================
# Report Generation Models
# ============================================================================

class ReportGenerationRequest(BaseModel):
    """Request to generate a report"""
    case_id: str
    report_type: str = "comprehensive"  # comprehensive, summary, legal_brief
    include_remedies: bool = True
    include_timeline: bool = True
    include_similar_cases: bool = True
    format: str = "pdf"  # pdf, docx, html
    style: str = "formal"  # formal, casual
    privacy_profile: str = "personal_identifiers"


# ============================================================================
# Audit Models
# ============================================================================

class AuditEventItem(BaseModel):
    id: int
    actor: str
    actor_user_id: Optional[int] = None
    action: str
    resource: str
    case_id: Optional[int] = None
    occurred_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditEventListResponse(BaseModel):
    case_id: int
    total: int
    events: List[AuditEventItem]


class ReportGenerationResponse(BaseModel):
    """Report generation response"""
    report_id: str
    job_id: str
    case_id: str
    status: str
    report_type: str
    format: str
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# ============================================================================
# Analytics Models
# ============================================================================

class CostBreakdown(BaseModel):
    """Cost breakdown for user"""
    period: str  # monthly, all_time
    total_cost: float
    llm_api_cost: float
    document_processing_cost: float
    storage_cost: float
    api_calls: int
    documents_analyzed: int
    reports_generated: int


class AnalyticsResponse(BaseModel):
    """Analytics data"""
    user_id: str
    cost_breakdown: CostBreakdown
    active_cases: int
    pending_deadlines: int
    successful_analyses: int
    failed_analyses: int
    average_analysis_time_seconds: float
    top_case_types: List[tuple]  # [(case_type, count), ...]
    generated_at: datetime


class DashboardSummaryResponse(BaseModel):
    """Dashboard summary for frontend consumers."""
    total_cases_processed: int
    appeals_filed: int
    appeal_rate_percent: float
    plaintiff_wins: int
    defendant_wins: int
    settlements: int
    dismissals: int


# ============================================================================
# Deadline Models
# ============================================================================

class DeadlineResponse(BaseModel):
    """User deadline"""
    deadline_id: str
    user_id: str
    case_id: Optional[str] = None
    title: str
    description: str
    due_date: datetime
    days_until_due: int
    priority: str  # low, medium, high, critical
    status: str  # pending, completed, overdue
    reminder_enabled: bool
    reminder_days: int
    created_at: datetime


class UpcomingDeadlinesResponse(BaseModel):
    """List of upcoming deadlines"""
    user_id: str
    total_deadlines: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    deadlines: List[DeadlineResponse]
    generated_at: datetime


# ============================================================================
# Knowledge Freshness Models
# ============================================================================

class KnowledgeInvalidationItem(BaseModel):
    id: int
    user_id: Optional[int] = None
    case_id: Optional[int] = None
    document_id: Optional[int] = None
    scope_type: str
    scope_value: str
    reason: str
    details: Optional[Dict[str, Any]] = None
    status: str
    invalidated_at: datetime
    scheduled_for: Optional[datetime] = None
    recompute_started_at: Optional[datetime] = None
    recompute_completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    recompute_attempts: int = 0


class KnowledgeInvalidationListResponse(BaseModel):
    items: List[KnowledgeInvalidationItem]
    total: int
    stale_count: int
    fresh_count: int
    next_recompute_at: Optional[datetime] = None
    generated_at: datetime


# ============================================================================
# User Models
# ============================================================================

class UserProfile(BaseModel):
    """User profile"""
    user_id: str
    email: EmailStr
    full_name: str
    organization: Optional[str] = None
    role: str = "user"  # user, attorney, admin
    subscription_tier: str = "free"  # free, pro, enterprise
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ValidationError(ErrorResponse):
    """Validation error"""
    error_code: str = "VALIDATION_ERROR"
    errors: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# Pagination Models
# ============================================================================

class PaginationParams(BaseModel):
    """Pagination parameters"""
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper"""
    total: int
    limit: int
    offset: int
    items: List[Dict[str, Any]]
