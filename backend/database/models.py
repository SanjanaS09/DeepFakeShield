"""
Database Models for Multi-Modal Deepfake Detection System
Comprehensive ORM models using SQLAlchemy for all system entities
Supports PostgreSQL, MySQL, and SQLite with proper indexing and relationships
"""

import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Table, Column, Integer, String, Text, Boolean, Float, DateTime,
    ForeignKey, Enum, JSON, LargeBinary, Index, UniqueConstraint,
    CheckConstraint, func
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func as sql_func
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as pgUUID

# Custom GUID type for database agnosticism
class GUID(TypeDecorator):
    """
    Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36) for other databases.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(pgUUID())
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value

# Create the declarative base
Base = declarative_base()

# Enum definitions
class UserRole(enum.Enum):
    USER = "user"
    ADMIN = "admin"
    ANALYST = "analyst"
    API_USER = "api_user"

class MediaType(enum.Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

class ProcessingStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DetectionDecision(enum.Enum):
    REAL = "real"
    DEEPFAKE = "deepfake"
    UNCERTAIN = "uncertain"

class ModelStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    TESTING = "testing"

class LogLevel(enum.Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class User(Base):
    __tablename__ = 'users'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(Enum(UserRole), nullable=False, default=UserRole.USER)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    api_key = Column(String(64), unique=True, index=True)
    api_quota_daily = Column(Integer, default=100)
    api_calls_today = Column(Integer, default=0)
    api_calls_total = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    last_api_call = Column(DateTime(timezone=True))
    user_metadata = Column(JSON, nullable=True)
    detection_requests = relationship("DetectionRequest", back_populates="user")
    analysis_requests = relationship("AnalysisRequest", back_populates="user")
    explanation_requests = relationship("ExplanationRequest", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    __table_args__ = (Index('idx_user_role_active', 'role', 'is_active'), Index('idx_user_created', 'created_at'))

class MediaFile(Base):
    __tablename__ = 'media_files'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    mime_type = Column(String(100))
    media_type = Column(Enum(MediaType), nullable=False, index=True)
    width = Column(Integer)
    height = Column(Integer)
    duration = Column(Float)
    fps = Column(Float)
    channels = Column(Integer)
    sample_rate = Column(Integer)
    bit_rate = Column(Integer)
    is_processed = Column(Boolean, default=False, nullable=False)
    processing_errors = Column(Text)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    user_metadata = Column(JSON, nullable=True)
    detection_requests = relationship("DetectionRequest", back_populates="media_file")
    analysis_requests = relationship("AnalysisRequest", back_populates="media_file")
    __table_args__ = (Index('idx_media_hash_type', 'file_hash', 'media_type'), Index('idx_media_created', 'created_at'), Index('idx_media_processed', 'is_processed'))

class Model(Base):
    __tablename__ = 'models'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(20), nullable=False)
    model_type = Column(String(50), nullable=False)
    modality = Column(Enum(MediaType), nullable=False)
    architecture = Column(String(100))
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    file_hash = Column(String(64))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_score = Column(Float)
    status = Column(Enum(ModelStatus), default=ModelStatus.TESTING, nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)
    priority = Column(Integer, default=0)
    training_dataset = Column(String(200))
    training_date = Column(DateTime(timezone=True))
    training_duration = Column(Float)
    training_parameters = Column(JSON)
    deployed_at = Column(DateTime(timezone=True))
    last_used = Column(DateTime(timezone=True))
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    user_metadata = Column(JSON, nullable=True)
    detection_results = relationship("DetectionResult", secondary="detection_result_models", back_populates="models")
    __table_args__ = (Index('idx_model_type_modality', 'model_type', 'modality'), Index('idx_model_status_priority', 'status', 'priority'), UniqueConstraint('name', 'version', name='uq_model_name_version'))

class DetectionRequest(Base):
    __tablename__ = 'detection_requests'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False)
    media_file_id = Column(GUID(), ForeignKey('media_files.id'), nullable=False, index=True)
    model_types = Column(JSON)
    confidence_threshold = Column(Float, default=0.5)
    batch_size = Column(Integer, default=8)
    enable_xai = Column(Boolean, default=True)
    extract_features = Column(Boolean, default=True)
    quality_assessment = Column(Boolean, default=True)
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False, index=True)
    priority = Column(Integer, default=0)
    submitted_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    processing_time = Column(Float)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    user_metadata = Column(JSON, nullable=True)
    client_info = Column(JSON)
    user = relationship("User", back_populates="detection_requests")
    media_file = relationship("MediaFile", back_populates="detection_requests")
    results = relationship("DetectionResult", back_populates="request", cascade="all, delete-orphan")
    explanation_requests = relationship("ExplanationRequest", back_populates="detection_request")
    __table_args__ = (Index('idx_detection_status_priority', 'status', 'priority'), Index('idx_detection_submitted', 'submitted_at'), Index('idx_detection_user_status', 'user_id', 'status'))

class DetectionResult(Base):
    __tablename__ = 'detection_results'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    request_id = Column(GUID(), ForeignKey('detection_requests.id'), nullable=False, index=True)
    decision = Column(Enum(DetectionDecision), nullable=False, index=True)
    confidence_score = Column(Float, nullable=False)
    is_deepfake = Column(Boolean, nullable=False, index=True)
    model_results = Column(JSON, nullable=False)
    aggregation_method = Column(String(50))
    processing_time = Column(Float, nullable=False)
    features_extracted = Column(Boolean, default=False)
    quality_assessed = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    user_metadata = Column(JSON, nullable=True)
    request = relationship("DetectionRequest", back_populates="results")
    features = relationship("Feature", back_populates="detection_result", cascade="all, delete-orphan")
    evidence = relationship("Evidence", back_populates="detection_result", cascade="all, delete-orphan")
    explanation_requests = relationship("ExplanationRequest", back_populates="detection_result")
    models = relationship("Model", secondary="detection_result_models", back_populates="detection_results")
    __table_args__ = (Index('idx_result_decision_confidence', 'decision', 'confidence_score'), Index('idx_result_created', 'created_at'), Index('idx_result_request_decision', 'request_id', 'decision'))

detection_result_models = Table(
    'detection_result_models', Base.metadata,
    Column('detection_result_id', GUID(), ForeignKey('detection_results.id'), primary_key=True),
    Column('model_id', GUID(), ForeignKey('models.id'), primary_key=True),
    Column('model_confidence', Float),
    Column('model_prediction', Integer),
    Column('created_at', DateTime(timezone=True), default=func.now())
)

class Feature(Base):
    __tablename__ = 'features'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    detection_result_id = Column(GUID(), ForeignKey('detection_results.id'), index=True)
    analysis_result_id = Column(GUID(), ForeignKey('analysis_results.id'), index=True)
    modality = Column(Enum(MediaType), nullable=False, index=True)
    feature_type = Column(String(100), nullable=False, index=True)
    feature_name = Column(String(200), nullable=False)
    feature_category = Column(String(100))
    feature_value = Column(Float)
    feature_vector = Column(JSON)
    feature_data = Column(JSON)
    extraction_method = Column(String(100))
    extraction_parameters = Column(JSON)
    quality_score = Column(Float)
    importance_score = Column(Float)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    detection_result = relationship("DetectionResult", back_populates="features")
    analysis_result = relationship("AnalysisResult", back_populates="features")
    __table_args__ = (Index('idx_feature_modality_type', 'modality', 'feature_type'), Index('idx_feature_importance', 'importance_score'), CheckConstraint('detection_result_id IS NOT NULL OR analysis_result_id IS NOT NULL'))

class Evidence(Base):
    __tablename__ = 'evidence'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    detection_result_id = Column(GUID(), ForeignKey('detection_results.id'), nullable=False, index=True)
    evidence_type = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=False)
    strength = Column(Float, nullable=False)
    supports = Column(String(50), nullable=False)
    evidence_value = Column(Float)
    evidence_data = Column(JSON)
    source_modality = Column(Enum(MediaType))
    source_feature = Column(String(200))
    detection_method = Column(String(100))
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    detection_result = relationship("DetectionResult", back_populates="evidence")
    __table_args__ = (Index('idx_evidence_type_strength', 'evidence_type', 'strength'), Index('idx_evidence_supports', 'supports'))

class AnalysisRequest(Base):
    __tablename__ = 'analysis_requests'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True)
    media_file_id = Column(GUID(), ForeignKey('media_files.id'), nullable=False, index=True)
    analysis_types = Column(JSON, nullable=False)
    output_format = Column(String(20), default='json')
    include_thumbnails = Column(Boolean, default=False)
    extract_frames = Column(Boolean, default=False)
    max_frames = Column(Integer, default=10)
    quality_assessment = Column(Boolean, default=True)
    face_analysis = Column(Boolean, default=True)
    artifact_detection = Column(Boolean, default=True)
    target_audience = Column(String(20), default='general')
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False, index=True)
    priority = Column(Integer, default=0)
    submitted_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    processing_time = Column(Float)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    user_metadata = Column(JSON, nullable=True)
    client_info = Column(JSON)
    user = relationship("User", back_populates="analysis_requests")
    media_file = relationship("MediaFile", back_populates="analysis_requests")
    results = relationship("AnalysisResult", back_populates="request", cascade="all, delete-orphan")
    __table_args__ = (Index('idx_analysis_status_priority', 'status', 'priority'), Index('idx_analysis_submitted', 'submitted_at'), Index('idx_analysis_user_status', 'user_id', 'status'))

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    request_id = Column(GUID(), ForeignKey('analysis_requests.id'), nullable=False, index=True)
    metadata_analysis = Column(JSON)
    quality_analysis = Column(JSON)
    content_analysis = Column(JSON)
    artifact_analysis = Column(JSON)
    statistical_analysis = Column(JSON)
    preprocessing_info = Column(JSON)
    thumbnails = Column(JSON)
    extracted_frames = Column(JSON)
    processing_time = Column(Float, nullable=False)
    warnings = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    user_metadata = Column(JSON, nullable=True)
    request = relationship("AnalysisRequest", back_populates="results")
    features = relationship("Feature", back_populates="analysis_result", cascade="all, delete-orphan")
    __table_args__ = (Index('idx_analysis_result_created', 'created_at'), Index('idx_analysis_result_processing_time', 'processing_time'))

class ExplanationRequest(Base):
    __tablename__ = 'explanation_requests'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True)
    detection_request_id = Column(GUID(), ForeignKey('detection_requests.id'), index=True)
    detection_result_id = Column(GUID(), ForeignKey('detection_results.id'), index=True)
    explanation_types = Column(JSON, nullable=False)
    confidence_threshold = Column(Float, default=0.5)
    generate_visualizations = Column(Boolean, default=True)
    include_evidence = Column(Boolean, default=True)
    target_audience = Column(String(20), default='general')
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False, index=True)
    priority = Column(Integer, default=0)
    submitted_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    processing_time = Column(Float)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    user_metadata = Column(JSON, nullable=True)
    client_info = Column(JSON)
    user = relationship("User", back_populates="explanation_requests")
    detection_request = relationship("DetectionRequest", back_populates="explanation_requests")
    detection_result = relationship("DetectionResult", back_populates="explanation_requests")
    results = relationship("ExplanationResult", back_populates="request", cascade="all, delete-orphan")
    __table_args__ = (Index('idx_explanation_status_priority', 'status', 'priority'), Index('idx_explanation_submitted', 'submitted_at'), CheckConstraint('detection_request_id IS NOT NULL OR detection_result_id IS NOT NULL'))

class ExplanationResult(Base):
    __tablename__ = 'explanation_results'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    request_id = Column(GUID(), ForeignKey('explanation_requests.id'), nullable=False, index=True)
    decision = Column(Enum(DetectionDecision), nullable=False)
    confidence = Column(Float, nullable=False)
    summary = Column(Text, nullable=False)
    detailed_explanation = Column(JSON)
    statistical_analysis = Column(JSON)
    visualizations = Column(JSON)
    evidence_items = Column(JSON)
    recommendations = Column(JSON)
    limitations = Column(JSON)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    user_metadata = Column(JSON, nullable=True)
    request = relationship("ExplanationRequest", back_populates="results")
    __table_args__ = (Index('idx_explanation_result_decision', 'decision'), Index('idx_explanation_result_created', 'created_at'))

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey('users.id'), index=True)
    event_type = Column(String(100), nullable=False, index=True)
    event_category = Column(String(50), nullable=False, index=True)
    level = Column(Enum(LogLevel), nullable=False, default=LogLevel.INFO, index=True)
    message = Column(Text, nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    action = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    session_id = Column(String(100))
    request_id = Column(String(100))
    event_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)
    user = relationship("User", back_populates="audit_logs")
    __table_args__ = (Index('idx_audit_event_type_level', 'event_type', 'level'), Index('idx_audit_user_created', 'user_id', 'created_at'), Index('idx_audit_resource', 'resource_type', 'resource_id'), Index('idx_audit_ip_created', 'ip_address', 'created_at'))

class SystemConfig(Base):
    __tablename__ = 'system_config'
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    config_key = Column(String(200), unique=True, nullable=False, index=True)
    config_value = Column(Text)
    config_type = Column(String(50), nullable=False)
    config_category = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    default_value = Column(Text)
    is_secret = Column(Boolean, default=False, nullable=False)
    is_editable = Column(Boolean, default=True, nullable=False)
    validation_rules = Column(JSON)
    version = Column(Integer, default=1, nullable=False)
    previous_value = Column(Text)
    changed_by = Column(GUID())
    changed_reason = Column(String(500))
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    __table_args__ = (Index('idx_config_category_key', 'config_category', 'config_key'), Index('idx_config_updated', 'updated_at'))

    def __repr__(self):
        return f"<SystemConfig(key='{self.config_key}', category='{self.config_category}')>"


# Additional utility functions for models

def create_detection_request(user_id, media_file_id, **kwargs):
    """Helper function to create a detection request"""
    request = DetectionRequest(
        user_id=user_id,
        media_file_id=media_file_id,
        **kwargs
    )
    return request

def create_analysis_request(user_id, media_file_id, analysis_types, **kwargs):
    """Helper function to create an analysis request"""
    request = AnalysisRequest(
        user_id=user_id,
        media_file_id=media_file_id,
        analysis_types=analysis_types,
        **kwargs
    )
    return request

def create_explanation_request(user_id, explanation_types, detection_result_id=None, 
                             detection_request_id=None, **kwargs):
    """Helper function to create an explanation request"""
    request = ExplanationRequest(
        user_id=user_id,
        explanation_types=explanation_types,
        detection_result_id=detection_result_id,
        detection_request_id=detection_request_id,
        **kwargs
    )
    return request

def get_active_models_by_modality(session, modality):
    """Get active models for a specific modality"""
    return session.query(Model).filter(
        Model.modality == modality,
        Model.status == ModelStatus.ACTIVE
    ).order_by(Model.priority.desc(), Model.created_at.desc()).all()

def log_audit_event(session, event_type, message, user_id=None, level=LogLevel.INFO, **kwargs):
    """Helper function to log audit events"""
    log_entry = AuditLog(
        user_id=user_id,
        event_type=event_type,
        level=level,
        message=message,
        **kwargs
    )
    session.add(log_entry)
    return log_entry

def get_user_api_usage(session, user_id, date=None):
    """Get API usage for a user"""
    if date is None:
        date = datetime.now(timezone.utc).date()

    # Count API calls for the day
    api_calls = session.query(AuditLog).filter(
        AuditLog.user_id == user_id,
        AuditLog.event_category == 'api',
        sql_func.date(AuditLog.created_at) == date
    ).count()

    return api_calls

def update_user_api_usage(session, user_id):
    """Update user's API usage counter"""
    user = session.query(User).filter(User.id == user_id).first()
    if user:
        user.api_calls_today = get_user_api_usage(session, user_id)
        user.api_calls_total += 1
        user.last_api_call = func.now()
        session.commit()

def get_system_config(session, config_key, default_value=None):
    """Get system configuration value"""
    config = session.query(SystemConfig).filter(
        SystemConfig.config_key == config_key
    ).first()

    if config:
        # Convert based on type
        if config.config_type == 'int':
            return int(config.config_value)
        elif config.config_type == 'float':
            return float(config.config_value)
        elif config.config_type == 'bool':
            return config.config_value.lower() in ('true', '1', 'yes')
        elif config.config_type == 'json':
            import json
            return json.loads(config.config_value)
        else:
            return config.config_value

    return default_value

def set_system_config(session, config_key, config_value, config_type='string', 
                     config_category='general', **kwargs):
    """Set system configuration value"""
    # Convert value to string for storage
    if config_type == 'json':
        import json
        value_str = json.dumps(config_value)
    else:
        value_str = str(config_value)

    config = session.query(SystemConfig).filter(
        SystemConfig.config_key == config_key
    ).first()

    if config:
        # Update existing config
        config.previous_value = config.config_value
        config.config_value = value_str
        config.version += 1
        config.updated_at = func.now()
        for key, value in kwargs.items():
            setattr(config, key, value)
    else:
        # Create new config
        config = SystemConfig(
            config_key=config_key,
            config_value=value_str,
            config_type=config_type,
            config_category=config_category,
            **kwargs
        )
        session.add(config)

    session.commit()
    return config
