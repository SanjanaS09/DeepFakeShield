"""
Database Utilities and Connection Management
Provides database connection, session management, and utility functions
Supports PostgreSQL, MySQL, and SQLite with connection pooling
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool, QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
import os
import logging
from typing import Optional, Dict, Any, Generator
from urllib.parse import quote_plus
import time

from .models import Base
from utils.logger import get_logger

logger = get_logger(__name__)

class DatabaseManager:
    """Database connection and session management"""

    def __init__(self, database_url: str = None, **engine_kwargs):
        """
        Initialize database manager

        Args:
            database_url: Database connection URL
            **engine_kwargs: Additional engine configuration
        """
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.SessionLocal = None
        self.Session = None
        self._engine_kwargs = engine_kwargs

        # Connection pool settings
        self.pool_settings = {
            'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
            'max_overflow': int(os.getenv('DB_POOL_OVERFLOW', '20')),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
            'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600')),
            'pool_pre_ping': True
        }

        self._initialize_engine()
        self._initialize_sessions()

    def _get_database_url(self) -> str:
        """Get database URL from environment variables"""

        # Check for full database URL first
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return database_url

        # Build URL from components
        db_type = os.getenv('DB_TYPE', 'sqlite')

        if db_type.lower() == 'sqlite':
            db_path = os.getenv('DB_PATH', 'deepfake_detection.db')
            return f'sqlite:///{db_path}'

        elif db_type.lower() == 'postgresql':
            user = os.getenv('DB_USER', 'postgres')
            password = quote_plus(os.getenv('DB_PASSWORD', ''))
            host = os.getenv('DB_HOST', 'localhost')
            port = os.getenv('DB_PORT', '5432')
            database = os.getenv('DB_NAME', 'deepfake_detection')

            return f'postgresql://{user}:{password}@{host}:{port}/{database}'

        elif db_type.lower() == 'mysql':
            user = os.getenv('DB_USER', 'root')
            password = quote_plus(os.getenv('DB_PASSWORD', ''))
            host = os.getenv('DB_HOST', 'localhost')
            port = os.getenv('DB_PORT', '3306')
            database = os.getenv('DB_NAME', 'deepfake_detection')

            return f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'

        else:
            raise ValueError(f"Unsupported database type: {db_type}")

    def _initialize_engine(self):
        """Initialize SQLAlchemy engine"""

        try:
            # Base engine configuration
            engine_config = {
                'echo': os.getenv('DB_ECHO', 'false').lower() == 'true',
                'future': True,
                **self._engine_kwargs
            }

            # Database-specific configuration
            if 'sqlite' in self.database_url:
                # SQLite configuration
                engine_config.update({
                    'poolclass': StaticPool,
                    'connect_args': {
                        'check_same_thread': False,
                        'timeout': 30,
                        'isolation_level': None
                    }
                })
            else:
                # PostgreSQL/MySQL configuration
                engine_config.update({
                    'poolclass': QueuePool,
                    **self.pool_settings
                })

            self.engine = create_engine(self.database_url, **engine_config)

            # Add event listeners
            self._add_event_listeners()

            logger.info(f"Database engine initialized: {self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url}")

        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    def _add_event_listeners(self):
        """Add SQLAlchemy event listeners for monitoring and optimization"""

        # Connection event listeners
        @event.listens_for(self.engine, "connect")
        def set_connection_settings(dbapi_connection, connection_record):
            """Set connection-specific settings"""
            try:
                if 'sqlite' in self.database_url:
                    # SQLite optimizations
                    dbapi_connection.execute("PRAGMA journal_mode=WAL")
                    dbapi_connection.execute("PRAGMA synchronous=NORMAL")
                    dbapi_connection.execute("PRAGMA cache_size=10000")
                    dbapi_connection.execute("PRAGMA temp_store=memory")
                    dbapi_connection.execute("PRAGMA mmap_size=268435456")  # 256MB

                elif 'postgresql' in self.database_url:
                    # PostgreSQL optimizations
                    dbapi_connection.autocommit = False
                    with dbapi_connection.cursor() as cursor:
                        cursor.execute("SET timezone = 'UTC'")
                        cursor.execute("SET statement_timeout = '300s'")

                elif 'mysql' in self.database_url:
                    # MySQL optimizations
                    dbapi_connection.autocommit = False
                    with dbapi_connection.cursor() as cursor:
                        cursor.execute("SET time_zone = '+00:00'")
                        cursor.execute("SET sql_mode = 'STRICT_TRANS_TABLES'")

            except Exception as e:
                logger.warning(f"Could not set connection settings: {e}")

        @event.listens_for(self.engine, "checkout")
        def checkout_listener(dbapi_connection, connection_record, connection_proxy):
            """Connection checkout event"""
            connection_record.info['checkout_time'] = time.time()

        @event.listens_for(self.engine, "checkin")
        def checkin_listener(dbapi_connection, connection_record):
            """Connection checkin event"""
            if 'checkout_time' in connection_record.info:
                checkout_time = connection_record.info.pop('checkout_time')
                connection_duration = time.time() - checkout_time
                if connection_duration > 60:  # Log long-running connections
                    logger.warning(f"Long-running database connection: {connection_duration:.2f}s")

    def _initialize_sessions(self):
        """Initialize session factories"""

        try:
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False
            )

            # Thread-safe scoped session
            self.Session = scoped_session(self.SessionLocal)

            logger.info("Database sessions initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database sessions: {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator:
        """
        Get database session with automatic cleanup

        Yields:
            Database session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    @contextmanager
    def get_transaction(self) -> Generator:
        """
        Get database session with explicit transaction management

        Yields:
            Database session
        """
        session = self.SessionLocal()
        transaction = session.begin()
        try:
            yield session
            transaction.commit()
        except Exception as e:
            transaction.rollback()
            logger.error(f"Database transaction error: {e}")
            raise
        finally:
            session.close()

    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def drop_tables(self):
        """Drop all database tables"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        try:
            info = {
                'url': self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url,
                'engine': str(self.engine.name),
                'pool_size': getattr(self.engine.pool, 'size', None),
                'pool_checked_out': getattr(self.engine.pool, 'checkedout', None),
                'pool_overflow': getattr(self.engine.pool, 'overflow', None),
                'echo': self.engine.echo
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {}

    def cleanup(self):
        """Cleanup database resources"""
        try:
            if self.Session:
                self.Session.remove()
            if self.engine:
                self.engine.dispose()
            logger.info("Database resources cleaned up")
        except Exception as e:
            logger.error(f"Database cleanup error: {e}")


# Global database manager instance
db_manager = None

def initialize_database(database_url: str = None, **kwargs) -> DatabaseManager:
    """Initialize global database manager"""
    global db_manager
    db_manager = DatabaseManager(database_url, **kwargs)
    return db_manager

def get_db_manager() -> DatabaseManager:
    """Get global database manager"""
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return db_manager

@contextmanager
def get_db_session():
    """Get database session from global manager"""
    manager = get_db_manager()
    with manager.get_session() as session:
        yield session

@contextmanager  
def get_db_transaction():
    """Get database transaction from global manager"""
    manager = get_db_manager()
    with manager.get_transaction() as session:
        yield session

def create_all_tables():
    """Create all database tables using global manager"""
    manager = get_db_manager()
    manager.create_tables()

def drop_all_tables():
    """Drop all database tables using global manager"""
    manager = get_db_manager()
    manager.drop_tables()

def test_db_connection() -> bool:
    """Test database connection using global manager"""
    manager = get_db_manager()
    return manager.test_connection()


class DatabaseHealthCheck:
    """Database health check utility"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def check_connection(self) -> Dict[str, Any]:
        """Check database connection health"""
        result = {
            'status': 'unknown',
            'response_time_ms': 0,
            'error': None
        }

        start_time = time.time()
        try:
            with self.db_manager.get_session() as session:
                session.execute("SELECT 1")

            response_time = (time.time() - start_time) * 1000
            result.update({
                'status': 'healthy',
                'response_time_ms': round(response_time, 2)
            })

        except Exception as e:
            result.update({
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': round((time.time() - start_time) * 1000, 2)
            })

        return result

    def check_tables(self) -> Dict[str, Any]:
        """Check if all required tables exist"""
        result = {
            'status': 'unknown',
            'tables_exist': [],
            'tables_missing': [],
            'error': None
        }

        required_tables = [
            'users', 'media_files', 'models', 'detection_requests',
            'detection_results', 'analysis_requests', 'analysis_results',
            'explanation_requests', 'explanation_results', 'features',
            'evidence', 'audit_logs', 'system_config'
        ]

        try:
            # Get list of existing tables
            inspector = self.db_manager.engine.inspect(self.db_manager.engine)
            existing_tables = inspector.get_table_names()

            tables_exist = []
            tables_missing = []

            for table in required_tables:
                if table in existing_tables:
                    tables_exist.append(table)
                else:
                    tables_missing.append(table)

            result.update({
                'status': 'healthy' if not tables_missing else 'unhealthy',
                'tables_exist': tables_exist,
                'tables_missing': tables_missing
            })

        except Exception as e:
            result.update({
                'status': 'unhealthy',
                'error': str(e)
            })

        return result

    def check_pool_status(self) -> Dict[str, Any]:
        """Check database connection pool status"""
        result = {
            'status': 'unknown',
            'pool_info': {},
            'error': None
        }

        try:
            pool = self.db_manager.engine.pool

            pool_info = {
                'size': getattr(pool, 'size', 'N/A'),
                'checked_out': getattr(pool, 'checkedout', 'N/A'),
                'overflow': getattr(pool, 'overflow', 'N/A'),
                'invalid': getattr(pool, 'invalid', 'N/A')
            }

            # Determine health status
            checked_out = getattr(pool, 'checkedout', 0)
            pool_size = getattr(pool, 'size', 10)

            if isinstance(checked_out, int) and isinstance(pool_size, int):
                utilization = checked_out / pool_size if pool_size > 0 else 0
                if utilization > 0.9:
                    status = 'warning'  # High utilization
                elif utilization > 0.95:
                    status = 'unhealthy'  # Very high utilization
                else:
                    status = 'healthy'
            else:
                status = 'healthy'  # Default for pools without these metrics

            result.update({
                'status': status,
                'pool_info': pool_info
            })

        except Exception as e:
            result.update({
                'status': 'unhealthy',
                'error': str(e)
            })

        return result

    def run_full_check(self) -> Dict[str, Any]:
        """Run comprehensive database health check"""

        health_report = {
            'overall_status': 'unknown',
            'timestamp': time.time(),
            'checks': {}
        }

        # Run individual checks
        checks = {
            'connection': self.check_connection,
            'tables': self.check_tables,
            'pool': self.check_pool_status
        }

        statuses = []

        for check_name, check_func in checks.items():
            try:
                check_result = check_func()
                health_report['checks'][check_name] = check_result
                statuses.append(check_result['status'])
            except Exception as e:
                health_report['checks'][check_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                statuses.append('unhealthy')

        # Determine overall status
        if all(status == 'healthy' for status in statuses):
            health_report['overall_status'] = 'healthy'
        elif any(status == 'unhealthy' for status in statuses):
            health_report['overall_status'] = 'unhealthy'
        else:
            health_report['overall_status'] = 'warning'

        return health_report


def get_health_check() -> DatabaseHealthCheck:
    """Get database health check utility"""
    manager = get_db_manager()
    return DatabaseHealthCheck(manager)


class DatabaseMigration:
    """Database migration utility"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def get_current_version(self) -> str:
        """Get current database version"""
        try:
            with self.db_manager.get_session() as session:
                result = session.execute(
                    "SELECT version_num FROM alembic_version ORDER BY version_num DESC LIMIT 1"
                )
                row = result.fetchone()
                return row[0] if row else None
        except Exception:
            return None

    def run_migrations(self):
        """Run database migrations using Alembic"""
        try:
            from alembic.config import Config
            from alembic import command

            # Create Alembic configuration
            alembic_cfg = Config()
            alembic_cfg.set_main_option("script_location", "database/migrations")
            alembic_cfg.set_main_option("sqlalchemy.url", self.db_manager.database_url)

            # Run migrations
            command.upgrade(alembic_cfg, "head")
            logger.info("Database migrations completed successfully")

        except ImportError:
            logger.warning("Alembic not available, creating tables directly")
            self.db_manager.create_tables()
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            raise

    def create_migration(self, message: str):
        """Create new migration"""
        try:
            from alembic.config import Config
            from alembic import command

            alembic_cfg = Config()
            alembic_cfg.set_main_option("script_location", "database/migrations")
            alembic_cfg.set_main_option("sqlalchemy.url", self.db_manager.database_url)

            command.revision(alembic_cfg, message=message, autogenerate=True)
            logger.info(f"Migration created: {message}")

        except ImportError:
            logger.error("Alembic not available, cannot create migration")
            raise
        except Exception as e:
            logger.error(f"Migration creation failed: {e}")
            raise


def get_migration_manager() -> DatabaseMigration:
    """Get database migration manager"""
    manager = get_db_manager()
    return DatabaseMigration(manager)


# Database initialization for different environments

def setup_development_db():
    """Setup database for development environment"""
    db_url = os.getenv('DATABASE_URL', 'sqlite:///deepfake_detection_dev.db')

    manager = initialize_database(
        database_url=db_url,
        echo=True,  # Enable SQL logging in development
    )

    # Create tables if they don't exist
    if not manager.test_connection():
        raise RuntimeError("Could not connect to development database")

    # Run migrations
    migration_manager = DatabaseMigration(manager)
    migration_manager.run_migrations()

    logger.info("Development database setup completed")
    return manager

def setup_production_db():
    """Setup database for production environment"""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is required for production")

    manager = initialize_database(
        database_url=db_url,
        echo=False,  # Disable SQL logging in production
        pool_size=20,
        max_overflow=30,
        pool_timeout=30,
        pool_recycle=3600
    )

    # Test connection
    if not manager.test_connection():
        raise RuntimeError("Could not connect to production database")

    # Run migrations
    migration_manager = DatabaseMigration(manager)
    migration_manager.run_migrations()

    logger.info("Production database setup completed")
    return manager

def setup_test_db():
    """Setup database for testing environment"""
    db_url = os.getenv('TEST_DATABASE_URL', 'sqlite:///deepfake_detection_test.db')

    manager = initialize_database(
        database_url=db_url,
        echo=False,
    )

    # Create tables directly for testing
    manager.create_tables()

    logger.info("Test database setup completed")
    return manager
