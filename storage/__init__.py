"""
storage package — Storage Adapters and Database Managers
=========================================================
AgentLens v2.0 pluggable storage layer.
"""

from storage.base_adapter import BaseStorageAdapter, get_storage_adapter
from storage.db import DatabaseManager
from storage.postgres_adapter import PostgreSQLStorageAdapter
from storage.sqlite_adapter import SQLiteStorageAdapter
from storage.writer import StorageWriter

__all__ = [
    "BaseStorageAdapter",
    "DatabaseManager",
    "PostgreSQLStorageAdapter",
    "SQLiteStorageAdapter",
    "StorageWriter",
    "get_storage_adapter",
]
