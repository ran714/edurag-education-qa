import csv
import time
from pathlib import Path

import pymysql

from base.config import Config
from base.knowledge_data import knowledge_csv

ROOT = Path(__file__).resolve().parent
CSV_PATH = knowledge_csv()


def connect(retries=30):
    settings = Config()
    last = None
    for _ in range(retries):
        try:
            return pymysql.connect(
                host=settings.MYSQL_HOST,
                port=settings.MYSQL_PORT,
                user=settings.MYSQL_USER,
                password=settings.MYSQL_PASSWORD,
                database=settings.MYSQL_DATABASE,
                charset="utf8mb4",
                autocommit=False,
            )
        except pymysql.MySQLError as exc:
            last = exc
            time.sleep(2)
    raise RuntimeError(f"MySQL 连接失败: {last}")


def main():
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jpkb (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    subject_name VARCHAR(50) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    FULLTEXT KEY idx_question_answer (question, answer)
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
            """)
            cursor.execute("SELECT COUNT(*) FROM jpkb")
            count = cursor.fetchone()[0]
            if count == 0 and CSV_PATH.exists():
                with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
                    rows = list(csv.DictReader(file))
                cursor.executemany(
                    "INSERT INTO jpkb(subject_name, question, answer) VALUES (%s, %s, %s)",
                    [(row["学科名称"], row["问题"], row["答案"]) for row in rows],
                )
                print(f"已导入 MySQL 知识问答 {len(rows)} 条")
            else:
                print(f"MySQL 知识问答已存在 {count} 条，跳过重复导入")
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
