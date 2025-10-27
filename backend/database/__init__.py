"""
Database package for session management, engine creation, and model initialization.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base
import os

# Get the database URL from environment variables, with a fallback to a local SQLite DB
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./deepfake_detection.db")

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    # connect_args is needed only for SQLite
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Creates all database tables based on the models defined in models.py.
    """
    print("Initializing database tables...")
    # The magic happens here: SQLAlchemy creates tables from all classes that use Base
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")

# Expose the key components to be used by the rest of the application
__all__ = ["engine", "SessionLocal", "init_db", "Base"]
