"""Database connection and session management."""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

from backend.core.config import settings
from backend.data.models import Base


class Database:
    """Database connection manager."""

    def __init__(self):
        """Initialize database connection."""
        self.engine = None
        self.SessionLocal = None
        self._initialized = False

    def initialize(self):
        """Create database engine and session factory."""
        if self._initialized:
            logger.warning("Database already initialized")
            return

        logger.info(f"Initializing database: {settings.database_url}")
        
        # Create engine
        if settings.database_url.startswith("sqlite"):
            self.engine = create_engine(
                settings.database_url,
                connect_args={"check_same_thread": False},
                echo=settings.debug,
            )
        else:
            self.engine = create_engine(
                settings.database_url,
                pool_pre_ping=True,
                echo=settings.debug,
            )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        self._initialized = True
        logger.info("Database initialized successfully")

    def create_tables(self):
        """Create all database tables."""
        if not self._initialized:
            self.initialize()

        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def drop_tables(self):
        """Drop all database tables (use with caution!)."""
        if not self._initialized:
            self.initialize()

        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")

    def get_session(self) -> Generator[Session, None, None]:
        """
        Get database session.

        Yields:
            Session: SQLAlchemy session

        Usage:
            with database.get_session() as session:
                # Use session
                pass
        """
        if not self._initialized:
            self.initialize()

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

    def close(self):
        """Close database engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global database instance
database = Database()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI to get database session.

    Yields:
        Session: SQLAlchemy session
    """
    return database.get_session()
