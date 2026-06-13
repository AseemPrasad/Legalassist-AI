import datetime as dt
import enum

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from db.base import Base


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    ATTORNEY = "attorney"
    PARALEGAL = "paralegal"
    CLIENT = "client"


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    role = Column(SQLEnum(UserRole), default=UserRole.CLIENT, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    is_verified = Column(Boolean, default=True, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    sso_provider = Column(String(50), nullable=True, index=True)
    sso_provider_id = Column(String(255), nullable=True)

    cases = relationship("Case", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    case_comments = relationship("CaseComment", back_populates="user", cascade="all, delete-orphan")
    case_presence = relationship("CasePresence", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role.value if self.role else "client",
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "is_verified": self.is_verified,
            "is_admin": self.is_admin,
        }


class OTPVerification(Base):
    __tablename__ = "otp_verifications"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False, index=True)
    otp_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    failed_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)

    def is_locked(self) -> bool:
        if self.locked_until is None:
            return False
        now = dt.datetime.now(dt.timezone.utc)
        locked_until = self.locked_until
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=dt.timezone.utc)
        return now < locked_until


class APIKey(Base):
    """Database model for API keys."""
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True)
    key_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(64), nullable=False)
    key_salt = Column(String(32), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    def is_valid(self) -> bool:
        if not self.expires_at:
            return True
        expires_at = self.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=dt.timezone.utc)
        now = dt.datetime.now(dt.timezone.utc)
        return now <= expires_at
