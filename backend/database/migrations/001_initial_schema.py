"""
Initial Database Migration for Multi-Modal Deepfake Detection System
Creates all tables and indexes with proper relationships and constraints
Compatible with PostgreSQL, MySQL, and SQLite backends

Revision ID: 001_initial_schema
Revises: None
Create Date: 2024-01-01 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, mysql
from sqlalchemy import text
import uuid

# revision identifiers
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create all tables and initial data"""

    # Create UUID extension for PostgreSQL
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('username', sa.String(50), nullable=False, unique=True),
        sa.Column('email', sa.String(100), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(100)),
        sa.Column('role', sa.Enum('USER', 'ADMIN', 'ANALYST', 'API_USER', name='userrole'), nullable=False, default='USER'),
        sa.Column('is_active', sa.Boolean, nullable=False, default=True),
        sa.Column('is_verified', sa.Boolean, nullable=False, default=False),
        sa.Column('api_key', sa.String(64), unique=True),
        sa.Column('api_quota_daily', sa.Integer, default=100),
        sa.Column('api_calls_today', sa.Integer, default=0),
        sa.Column('api_calls_total', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('last_login', sa.DateTime(timezone=True)),
        sa.Column('last_api_call', sa.DateTime(timezone=True)),
        sa.Column('metadata', sa.JSON)
    )

    # Create indexes for users
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_api_key', 'users', ['api_key'])
    op.create_index('idx_user_role_active', 'users', ['role', 'is_active'])
    op.create_index('idx_user_created', 'users', ['created_at'])

    # Media files table
    op.create_table(
        'media_files',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('filename', sa.String(255), nullable=False),
        sa.Column('original_filename', sa.String(255)),
        sa.Column('file_path', sa.String(500), nullable=False),
        sa.Column('file_size', sa.Integer, nullable=False),
        sa.Column('file_hash', sa.String(64), nullable=False),
        sa.Column('mime_type', sa.String(100)),
        sa.Column('media_type', sa.Enum('IMAGE', 'VIDEO', 'AUDIO', 'MULTIMODAL', name='mediatype'), nullable=False),
        sa.Column('width', sa.Integer),
        sa.Column('height', sa.Integer),
        sa.Column('duration', sa.Float),
        sa.Column('fps', sa.Float),
        sa.Column('channels', sa.Integer),
        sa.Column('sample_rate', sa.Integer),
        sa.Column('bit_rate', sa.Integer),
        sa.Column('is_processed', sa.Boolean, nullable=False, default=False),
        sa.Column('processing_errors', sa.Text),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', sa.JSON)
    )

    # Create indexes for media files
    op.create_index('idx_media_hash_type', 'media_files', ['file_hash', 'media_type'])
    op.create_index('idx_media_created', 'media_files', ['created_at'])
    op.create_index('idx_media_processed', 'media_files', ['is_processed'])
    op.create_index('idx_media_type', 'media_files', ['media_type'])

    # Models table
    op.create_table(
        'models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('version', sa.String(20), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),
        sa.Column('modality', sa.Enum('IMAGE', 'VIDEO', 'AUDIO', 'MULTIMODAL', name='mediatype'), nullable=False),
        sa.Column('architecture', sa.String(100)),
        sa.Column('file_path', sa.String(500), nullable=False),
        sa.Column('file_size', sa.Integer),
        sa.Column('file_hash', sa.String(64)),
        sa.Column('accuracy', sa.Float),
        sa.Column('precision', sa.Float),
        sa.Column('recall', sa.Float),
        sa.Column('f1_score', sa.Float),
        sa.Column('auc_score', sa.Float),
        sa.Column('status', sa.Enum('ACTIVE', 'INACTIVE', 'DEPRECATED', 'TESTING', name='modelstatus'), nullable=False, default='TESTING'),
        sa.Column('is_default', sa.Boolean, nullable=False, default=False),
        sa.Column('priority', sa.Integer, default=0),
        sa.Column('training_dataset', sa.String(200)),
        sa.Column('training_date', sa.DateTime(timezone=True)),
        sa.Column('training_duration', sa.Float),
        sa.Column('training_parameters', sa.JSON),
        sa.Column('deployed_at', sa.DateTime(timezone=True)),
        sa.Column('last_used', sa.DateTime(timezone=True)),
        sa.Column('usage_count', sa.Integer, default=0),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('metadata', sa.JSON)
    )

    # Create indexes for models
    op.create_index('idx_models_name', 'models', ['name'])
    op.create_index('idx_model_type_modality', 'models', ['model_type', 'modality'])
    op.create_index('idx_model_status_priority', 'models', ['status', 'priority'])
    op.create_unique_constraint('uq_model_name_version', 'models', ['name', 'version'])

    # Detection requests table
    op.create_table(
        'detection_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('media_file_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('media_files.id'), nullable=False),
        sa.Column('model_types', sa.JSON),
        sa.Column('confidence_threshold', sa.Float, default=0.5),
        sa.Column('batch_size', sa.Integer, default=8),
        sa.Column('enable_xai', sa.Boolean, default=True),
        sa.Column('extract_features', sa.Boolean, default=True),
        sa.Column('quality_assessment', sa.Boolean, default=True),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED', name='processingstatus'), nullable=False, default='PENDING'),
        sa.Column('priority', sa.Integer, default=0),
        sa.Column('submitted_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('processing_time', sa.Float),
        sa.Column('error_message', sa.Text),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('max_retries', sa.Integer, default=3),
        sa.Column('metadata', sa.JSON),
        sa.Column('client_info', sa.JSON)
    )

    # Create indexes for detection requests
    op.create_index('idx_detection_user', 'detection_requests', ['user_id'])
    op.create_index('idx_detection_media', 'detection_requests', ['media_file_id'])
    op.create_index('idx_detection_status_priority', 'detection_requests', ['status', 'priority'])
    op.create_index('idx_detection_submitted', 'detection_requests', ['submitted_at'])
    op.create_index('idx_detection_user_status', 'detection_requests', ['user_id', 'status'])

    # Detection results table
    op.create_table(
        'detection_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('detection_requests.id'), nullable=False),
        sa.Column('decision', sa.Enum('REAL', 'DEEPFAKE', 'UNCERTAIN', name='detectiondecision'), nullable=False),
        sa.Column('confidence_score', sa.Float, nullable=False),
        sa.Column('is_deepfake', sa.Boolean, nullable=False),
        sa.Column('model_results', sa.JSON, nullable=False),
        sa.Column('aggregation_method', sa.String(50)),
        sa.Column('processing_time', sa.Float, nullable=False),
        sa.Column('features_extracted', sa.Boolean, default=False),
        sa.Column('quality_assessed', sa.Boolean, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('metadata', sa.JSON)
    )

    # Create indexes for detection results
    op.create_index('idx_detection_results_request', 'detection_results', ['request_id'])
    op.create_index('idx_result_decision_confidence', 'detection_results', ['decision', 'confidence_score'])
    op.create_index('idx_result_created', 'detection_results', ['created_at'])
    op.create_index('idx_result_request_decision', 'detection_results', ['request_id', 'decision'])
    op.create_index('idx_result_deepfake', 'detection_results', ['is_deepfake'])

    # Analysis requests table
    op.create_table(
        'analysis_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('media_file_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('media_files.id'), nullable=False),
        sa.Column('analysis_types', sa.JSON, nullable=False),
        sa.Column('output_format', sa.String(20), default='json'),
        sa.Column('include_thumbnails', sa.Boolean, default=False),
        sa.Column('extract_frames', sa.Boolean, default=False),
        sa.Column('max_frames', sa.Integer, default=10),
        sa.Column('quality_assessment', sa.Boolean, default=True),
        sa.Column('face_analysis', sa.Boolean, default=True),
        sa.Column('artifact_detection', sa.Boolean, default=True),
        sa.Column('target_audience', sa.String(20), default='general'),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED', name='processingstatus'), nullable=False, default='PENDING'),
        sa.Column('priority', sa.Integer, default=0),
        sa.Column('submitted_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('processing_time', sa.Float),
        sa.Column('error_message', sa.Text),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('max_retries', sa.Integer, default=3),
        sa.Column('metadata', sa.JSON),
        sa.Column('client_info', sa.JSON)
    )

    # Create indexes for analysis requests
    op.create_index('idx_analysis_user', 'analysis_requests', ['user_id'])
    op.create_index('idx_analysis_media', 'analysis_requests', ['media_file_id'])
    op.create_index('idx_analysis_status_priority', 'analysis_requests', ['status', 'priority'])
    op.create_index('idx_analysis_submitted', 'analysis_requests', ['submitted_at'])
    op.create_index('idx_analysis_user_status', 'analysis_requests', ['user_id', 'status'])

    # Analysis results table
    op.create_table(
        'analysis_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analysis_requests.id'), nullable=False),
        sa.Column('metadata_analysis', sa.JSON),
        sa.Column('quality_analysis', sa.JSON),
        sa.Column('content_analysis', sa.JSON),
        sa.Column('artifact_analysis', sa.JSON),
        sa.Column('statistical_analysis', sa.JSON),
        sa.Column('preprocessing_info', sa.JSON),
        sa.Column('thumbnails', sa.JSON),
        sa.Column('extracted_frames', sa.JSON),
        sa.Column('processing_time', sa.Float, nullable=False),
        sa.Column('warnings', sa.JSON),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('metadata', sa.JSON)
    )

    # Create indexes for analysis results
    op.create_index('idx_analysis_results_request', 'analysis_results', ['request_id'])
    op.create_index('idx_analysis_result_created', 'analysis_results', ['created_at'])
    op.create_index('idx_analysis_result_processing_time', 'analysis_results', ['processing_time'])

    # Explanation requests table
    op.create_table(
        'explanation_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('detection_request_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('detection_requests.id')),
        sa.Column('detection_result_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('detection_results.id')),
        sa.Column('explanation_types', sa.JSON, nullable=False),
        sa.Column('confidence_threshold', sa.Float, default=0.5),
        sa.Column('generate_visualizations', sa.Boolean, default=True),
        sa.Column('include_evidence', sa.Boolean, default=True),
        sa.Column('target_audience', sa.String(20), default='general'),
        sa.Column('status', sa.Enum('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED', name='processingstatus'), nullable=False, default='PENDING'),
        sa.Column('priority', sa.Integer, default=0),
        sa.Column('submitted_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('processing_time', sa.Float),
        sa.Column('error_message', sa.Text),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('max_retries', sa.Integer, default=3),
        sa.Column('metadata', sa.JSON),
        sa.Column('client_info', sa.JSON)
    )

    # Create indexes for explanation requests
    op.create_index('idx_explanation_user', 'explanation_requests', ['user_id'])
    op.create_index('idx_explanation_detection_request', 'explanation_requests', ['detection_request_id'])
    op.create_index('idx_explanation_detection_result', 'explanation_requests', ['detection_result_id'])
    op.create_index('idx_explanation_status_priority', 'explanation_requests', ['status', 'priority'])
    op.create_index('idx_explanation_submitted', 'explanation_requests', ['submitted_at'])

    # Add check constraint for explanation requests
    op.create_check_constraint(
        'ck_explanation_request_source',
        'explanation_requests',
        'detection_request_id IS NOT NULL OR detection_result_id IS NOT NULL'
    )

    # Explanation results table
    op.create_table(
        'explanation_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('request_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('explanation_requests.id'), nullable=False),
        sa.Column('decision', sa.Enum('REAL', 'DEEPFAKE', 'UNCERTAIN', name='detectiondecision'), nullable=False),
        sa.Column('confidence', sa.Float, nullable=False),
        sa.Column('summary', sa.Text, nullable=False),
        sa.Column('detailed_explanation', sa.JSON),
        sa.Column('statistical_analysis', sa.JSON),
        sa.Column('visualizations', sa.JSON),
        sa.Column('evidence_items', sa.JSON),
        sa.Column('recommendations', sa.JSON),
        sa.Column('limitations', sa.JSON),
        sa.Column('processing_time', sa.Float, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('metadata', sa.JSON)
    )

    # Create indexes for explanation results
    op.create_index('idx_explanation_results_request', 'explanation_results', ['request_id'])
    op.create_index('idx_explanation_result_decision', 'explanation_results', ['decision'])
    op.create_index('idx_explanation_result_created', 'explanation_results', ['created_at'])

    # Features table
    op.create_table(
        'features',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('detection_result_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('detection_results.id')),
        sa.Column('analysis_result_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analysis_results.id')),
        sa.Column('modality', sa.Enum('IMAGE', 'VIDEO', 'AUDIO', 'MULTIMODAL', name='mediatype'), nullable=False),
        sa.Column('feature_type', sa.String(100), nullable=False),
        sa.Column('feature_name', sa.String(200), nullable=False),
        sa.Column('feature_category', sa.String(100)),
        sa.Column('feature_value', sa.Float),
        sa.Column('feature_vector', sa.JSON),
        sa.Column('feature_data', sa.JSON),
        sa.Column('extraction_method', sa.String(100)),
        sa.Column('extraction_parameters', sa.JSON),
        sa.Column('quality_score', sa.Float),
        sa.Column('importance_score', sa.Float),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )

    # Create indexes for features
    op.create_index('idx_features_detection_result', 'features', ['detection_result_id'])
    op.create_index('idx_features_analysis_result', 'features', ['analysis_result_id'])
    op.create_index('idx_feature_modality_type', 'features', ['modality', 'feature_type'])
    op.create_index('idx_feature_importance', 'features', ['importance_score'])

    # Add check constraint for features
    op.create_check_constraint(
        'ck_features_source',
        'features',
        'detection_result_id IS NOT NULL OR analysis_result_id IS NOT NULL'
    )

    # Evidence table
    op.create_table(
        'evidence',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('detection_result_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('detection_results.id'), nullable=False),
        sa.Column('evidence_type', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('strength', sa.Float, nullable=False),
        sa.Column('supports', sa.String(50), nullable=False),
        sa.Column('evidence_value', sa.Float),
        sa.Column('evidence_data', sa.JSON),
        sa.Column('source_modality', sa.Enum('IMAGE', 'VIDEO', 'AUDIO', 'MULTIMODAL', name='mediatype')),
        sa.Column('source_feature', sa.String(200)),
        sa.Column('detection_method', sa.String(100)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )

    # Create indexes for evidence
    op.create_index('idx_evidence_detection_result', 'evidence', ['detection_result_id'])
    op.create_index('idx_evidence_type_strength', 'evidence', ['evidence_type', 'strength'])
    op.create_index('idx_evidence_supports', 'evidence', ['supports'])

    # Detection result models association table
    op.create_table(
        'detection_result_models',
        sa.Column('detection_result_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('detection_results.id'), primary_key=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('models.id'), primary_key=True),
        sa.Column('model_confidence', sa.Float),
        sa.Column('model_prediction', sa.Integer),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    # Audit logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id')),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('event_category', sa.String(50), nullable=False),
        sa.Column('level', sa.Enum('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', name='loglevel'), nullable=False, default='INFO'),
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('resource_type', sa.String(100)),
        sa.Column('resource_id', sa.String(100)),
        sa.Column('action', sa.String(100)),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('user_agent', sa.String(500)),
        sa.Column('session_id', sa.String(100)),
        sa.Column('request_id', sa.String(100)),
        sa.Column('event_data', sa.JSON),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)
    )

    # Create indexes for audit logs
    op.create_index('idx_audit_user', 'audit_logs', ['user_id'])
    op.create_index('idx_audit_event_type_level', 'audit_logs', ['event_type', 'level'])
    op.create_index('idx_audit_user_created', 'audit_logs', ['user_id', 'created_at'])
    op.create_index('idx_audit_resource', 'audit_logs', ['resource_type', 'resource_id'])
    op.create_index('idx_audit_ip_created', 'audit_logs', ['ip_address', 'created_at'])
    op.create_index('idx_audit_created', 'audit_logs', ['created_at'])
    op.create_index('idx_audit_event_type', 'audit_logs', ['event_type'])
    op.create_index('idx_audit_level', 'audit_logs', ['level'])

    # System config table
    op.create_table(
        'system_config',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('config_key', sa.String(200), nullable=False, unique=True),
        sa.Column('config_value', sa.Text),
        sa.Column('config_type', sa.String(50), nullable=False),
        sa.Column('config_category', sa.String(100), nullable=False),
        sa.Column('description', sa.Text),
        sa.Column('default_value', sa.Text),
        sa.Column('is_secret', sa.Boolean, nullable=False, default=False),
        sa.Column('is_editable', sa.Boolean, nullable=False, default=True),
        sa.Column('validation_rules', sa.JSON),
        sa.Column('version', sa.Integer, nullable=False, default=1),
        sa.Column('previous_value', sa.Text),
        sa.Column('changed_by', postgresql.UUID(as_uuid=True)),
        sa.Column('changed_reason', sa.String(500)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Create indexes for system config
    op.create_index('idx_config_key', 'system_config', ['config_key'])
    op.create_index('idx_config_category_key', 'system_config', ['config_category', 'config_key'])
    op.create_index('idx_config_updated', 'system_config', ['updated_at'])
    op.create_index('idx_config_category', 'system_config', ['config_category'])

    # Insert initial system configuration
    _insert_initial_config()

    # Insert default admin user
    _insert_default_admin()


def downgrade():
    """Drop all tables in reverse order"""

    # Drop tables in reverse dependency order
    op.drop_table('system_config')
    op.drop_table('audit_logs')
    op.drop_table('detection_result_models')
    op.drop_table('evidence')
    op.drop_table('features')
    op.drop_table('explanation_results')
    op.drop_table('explanation_requests')
    op.drop_table('analysis_results')
    op.drop_table('analysis_requests')
    op.drop_table('detection_results')
    op.drop_table('detection_requests')
    op.drop_table('models')
    op.drop_table('media_files')
    op.drop_table('users')

    # Drop enums
    op.execute("DROP TYPE IF EXISTS userrole")
    op.execute("DROP TYPE IF EXISTS mediatype")
    op.execute("DROP TYPE IF EXISTS processingstatus")
    op.execute("DROP TYPE IF EXISTS detectiondecision")
    op.execute("DROP TYPE IF EXISTS modelstatus")
    op.execute("DROP TYPE IF EXISTS loglevel")


def _insert_initial_config():
    """Insert initial system configuration"""

    configs = [
        # Detection settings
        ('default_confidence_threshold', '0.5', 'float', 'detection', 'Default confidence threshold for deepfake detection'),
        ('max_file_size_mb', '100', 'int', 'detection', 'Maximum file size for uploads (MB)'),
        ('max_batch_size', '32', 'int', 'detection', 'Maximum batch size for processing'),
        ('enable_gpu', 'true', 'bool', 'detection', 'Enable GPU acceleration'),
        ('default_models_image', '["xception", "efficientnet"]', 'json', 'detection', 'Default models for image detection'),
        ('default_models_video', '["i3d", "slowfast"]', 'json', 'detection', 'Default models for video detection'),
        ('default_models_audio', '["ecapa_tdnn", "wav2vec2"]', 'json', 'detection', 'Default models for audio detection'),

        # Analysis settings
        ('enable_quality_assessment', 'true', 'bool', 'analysis', 'Enable quality assessment by default'),
        ('enable_face_analysis', 'true', 'bool', 'analysis', 'Enable face analysis by default'),
        ('enable_artifact_detection', 'true', 'bool', 'analysis', 'Enable artifact detection by default'),
        ('max_extracted_frames', '20', 'int', 'analysis', 'Maximum number of frames to extract from videos'),

        # API settings
        ('api_rate_limit_per_minute', '60', 'int', 'api', 'API rate limit per minute per user'),
        ('api_daily_quota_default', '1000', 'int', 'api', 'Default daily API quota for users'),
        ('api_require_authentication', 'true', 'bool', 'api', 'Require API authentication'),

        # Security settings
        ('session_timeout_minutes', '60', 'int', 'security', 'Session timeout in minutes'),
        ('max_login_attempts', '5', 'int', 'security', 'Maximum login attempts before lockout'),
        ('password_min_length', '8', 'int', 'security', 'Minimum password length'),
        ('require_email_verification', 'true', 'bool', 'security', 'Require email verification for new accounts'),

        # Storage settings
        ('media_storage_path', '/var/lib/deepfake_detection/media', 'string', 'storage', 'Path for media file storage'),
        ('model_storage_path', '/var/lib/deepfake_detection/models', 'string', 'storage', 'Path for model file storage'),
        ('temp_storage_path', '/tmp/deepfake_detection', 'string', 'storage', 'Path for temporary file storage'),
        ('max_storage_days', '30', 'int', 'storage', 'Maximum days to keep processed files'),

        # System settings
        ('system_name', 'Multi-Modal Deepfake Detection System', 'string', 'system', 'System display name'),
        ('system_version', '1.0.0', 'string', 'system', 'System version'),
        ('maintenance_mode', 'false', 'bool', 'system', 'System maintenance mode'),
        ('debug_mode', 'false', 'bool', 'system', 'Debug mode (development only)'),

        # Logging settings
        ('log_level', 'INFO', 'string', 'logging', 'System log level'),
        ('log_retention_days', '90', 'int', 'logging', 'Log retention period in days'),
        ('enable_audit_logging', 'true', 'bool', 'logging', 'Enable audit logging'),

        # Performance settings
        ('max_concurrent_requests', '10', 'int', 'performance', 'Maximum concurrent processing requests'),
        ('request_timeout_seconds', '300', 'int', 'performance', 'Request timeout in seconds'),
        ('enable_caching', 'true', 'bool', 'performance', 'Enable result caching'),
        ('cache_ttl_hours', '24', 'int', 'performance', 'Cache time-to-live in hours')
    ]

    # Insert configurations
    config_table = sa.table('system_config',
        sa.column('config_key', sa.String),
        sa.column('config_value', sa.Text),
        sa.column('config_type', sa.String),
        sa.column('config_category', sa.String),
        sa.column('description', sa.Text),
        sa.column('default_value', sa.Text),
        sa.column('is_secret', sa.Boolean),
        sa.column('is_editable', sa.Boolean)
    )

    for config_key, config_value, config_type, config_category, description in configs:
        op.bulk_insert(config_table, [{
            'config_key': config_key,
            'config_value': config_value,
            'config_type': config_type,
            'config_category': config_category,
            'description': description,
            'default_value': config_value,
            'is_secret': False,
            'is_editable': True
        }])


def _insert_default_admin():
    """Insert default admin user"""
    import hashlib
    import secrets

    # Create default admin user
    admin_password = 'admin123'  # Should be changed on first login
    password_hash = hashlib.sha256(admin_password.encode()).hexdigest()
    api_key = secrets.token_urlsafe(32)

    user_table = sa.table('users',
        sa.column('username', sa.String),
        sa.column('email', sa.String),
        sa.column('password_hash', sa.String),
        sa.column('full_name', sa.String),
        sa.column('role', sa.String),
        sa.column('is_active', sa.Boolean),
        sa.column('is_verified', sa.Boolean),
        sa.column('api_key', sa.String),
        sa.column('api_quota_daily', sa.Integer)
    )

    op.bulk_insert(user_table, [{
        'username': 'admin',
        'email': 'admin@deepfakedetection.local',
        'password_hash': password_hash,
        'full_name': 'System Administrator',
        'role': 'ADMIN',
        'is_active': True,
        'is_verified': True,
        'api_key': api_key,
        'api_quota_daily': 10000
    }])
