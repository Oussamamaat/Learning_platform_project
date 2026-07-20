import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    source_name = Column(String(500), nullable=False, index=True)
    source_type = Column(String(50), nullable=False, default="text")
    language = Column(String(10), nullable=False, default="fr")
    embedding = Column(Vector(384))
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Document(id={self.id}, source={self.source_name}, tenant={self.tenant_id})>"
