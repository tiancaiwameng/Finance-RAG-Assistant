"""可选 MySQL 元数据仓储；连接失败时自动降级为不记录模式。"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from config import Settings


logger = logging.getLogger(__name__)

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:  # 依赖未安装时仍允许应用以无数据库模式启动。
    mysql = None  # type: ignore[assignment]
    MySQLError = Exception  # type: ignore[misc,assignment]


class MySQLRepository:
    """保存文档处理状态和问答记录，不参与向量检索。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.mysql_enabled
        self.available = False

    @property
    def status_label(self) -> str:
        if not self.enabled:
            return "未启用（不记录）"
        return "已连接" if self.available else "连接失败（已降级）"

    def _connection_args(self, include_database: bool = True) -> dict[str, Any]:
        args: dict[str, Any] = {
            "host": self.settings.mysql_host,
            "port": self.settings.mysql_port,
            "user": self.settings.mysql_user,
            "password": self.settings.mysql_password,
            "connection_timeout": self.settings.mysql_connect_timeout,
            "charset": "utf8mb4",
        }
        if include_database:
            args["database"] = self.settings.mysql_database
        return args

    def initialize(self) -> bool:
        """按需创建数据库和表；任何失败都只触发降级，不影响主流程。"""
        if not self.enabled:
            logger.info("MySQL 未启用，使用不记录数据库模式")
            return False
        if mysql is None:
            logger.warning("未安装 mysql-connector-python，MySQL 记录功能已禁用")
            return False
        if not self.settings.mysql_user or not self.settings.mysql_database:
            logger.warning("MySQL 配置不完整，已降级为不记录数据库模式")
            return False
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.settings.mysql_database):
            logger.warning("MYSQL_DATABASE 只能包含字母、数字和下划线，MySQL 已降级")
            return False

        try:
            server_connection = mysql.connector.connect(**self._connection_args(False))
            try:
                cursor = server_connection.cursor()
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{self.settings.mysql_database}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
                cursor.close()
            finally:
                server_connection.close()

            connection = mysql.connector.connect(**self._connection_args())
            try:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                        filename VARCHAR(255) NOT NULL,
                        file_hash CHAR(64) NOT NULL,
                        upload_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        chunk_count INT UNSIGNED NOT NULL DEFAULT 0,
                        processing_status VARCHAR(32) NOT NULL,
                        PRIMARY KEY (id),
                        UNIQUE KEY uk_documents_file_hash (file_hash)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS qa_history (
                        id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                        document_id BIGINT UNSIGNED NOT NULL,
                        question TEXT NOT NULL,
                        answer MEDIUMTEXT NOT NULL,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (id),
                        KEY idx_qa_document_created (document_id, created_at),
                        CONSTRAINT fk_qa_document FOREIGN KEY (document_id)
                            REFERENCES documents (id) ON DELETE CASCADE
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                )
                connection.commit()
                cursor.close()
            finally:
                connection.close()
            self.available = True
            logger.info("MySQL 初始化完成：%s", self.settings.mysql_database)
        except MySQLError as exc:
            self.available = False
            logger.warning("MySQL 初始化失败，已降级为不记录数据库模式：%s", exc)
        return self.available

    def _handle_error(self, action: str, exc: Exception) -> None:
        self.available = False
        logger.warning("MySQL %s失败，后续操作已降级为不记录模式：%s", action, exc)

    def upsert_document(
        self, filename: str, file_hash: str, chunk_count: int, processing_status: str
    ) -> int | None:
        """按文件哈希新增或更新文档，并返回稳定的文档 ID。"""
        if not self.available or mysql is None:
            return None
        try:
            connection = mysql.connector.connect(**self._connection_args())
            try:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    INSERT INTO documents
                        (filename, file_hash, upload_time, chunk_count, processing_status)
                    VALUES (%s, %s, CURRENT_TIMESTAMP, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        id = LAST_INSERT_ID(id), filename = VALUES(filename),
                        upload_time = CURRENT_TIMESTAMP, chunk_count = VALUES(chunk_count),
                        processing_status = VALUES(processing_status)
                    """,
                    (filename, file_hash, chunk_count, processing_status),
                )
                document_id = int(cursor.lastrowid)
                connection.commit()
                cursor.close()
                return document_id
            finally:
                connection.close()
        except MySQLError as exc:
            self._handle_error("写入文档元数据", exc)
            return None

    def update_document(self, file_hash: str, chunk_count: int, status: str) -> None:
        """更新文档处理结果；数据库不可用时静默跳过。"""
        if not self.available or mysql is None:
            return
        try:
            connection = mysql.connector.connect(**self._connection_args())
            try:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    UPDATE documents
                    SET chunk_count = %s, processing_status = %s
                    WHERE file_hash = %s
                    """,
                    (chunk_count, status, file_hash),
                )
                connection.commit()
                cursor.close()
            finally:
                connection.close()
        except MySQLError as exc:
            self._handle_error("更新文档状态", exc)

    def record_qa(self, document_ids: Iterable[int], question: str, answer: str) -> None:
        """为答案实际引用的每个文档写入一条问答关联记录。"""
        if not self.available or mysql is None:
            return
        unique_ids = sorted(set(document_ids))
        if not unique_ids:
            return
        try:
            connection = mysql.connector.connect(**self._connection_args())
            try:
                cursor = connection.cursor()
                cursor.executemany(
                    "INSERT INTO qa_history (document_id, question, answer) VALUES (%s, %s, %s)",
                    [(document_id, question, answer) for document_id in unique_ids],
                )
                connection.commit()
                cursor.close()
                logger.info("问答记录已关联到 %d 个文档", len(unique_ids))
            finally:
                connection.close()
        except MySQLError as exc:
            self._handle_error("写入问答历史", exc)
