"""
Base model classes and common fields.
"""

from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.ext.declarative import declared_attr
from app.core.database import Base
import uuid


class BaseModel(Base):
    """Base model with common fields."""
    
    __abstract__ = True
    
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())