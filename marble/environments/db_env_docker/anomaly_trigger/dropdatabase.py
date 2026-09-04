import psycopg2
from utils.database import DB_CONFIG


def dropdatabase(name):
    # 连接到 "postgres" 数据库
    conn = psycopg2.connect(
        dbname="sysbench",  # 连接到默认的 "postgres" 数据库
        user=DB_CONFIG["user"],  # 替换为你的数据库用户名
        password=DB_CONFIG["password"],  # 替换为你的数据库密码
        host=DB_CONFIG["host"],  # 替换为你的数据库主机地址
        port=DB_CONFIG.get("port", 5432),
    )

    conn.autocommit = True
    # 创建一个数据库游标
    cur = conn.cursor()

    # 强制终结该数据库的所有活跃连接，避免被占用导致 drop 失败
    try:
        cur.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid();",
            (name,),
        )
    except Exception:
        pass

    # 删除数据库的 SQL 语句 (带 FORCE 支持)
    try:
        cur.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
    except Exception:
        cur.execute(f"DROP DATABASE IF EXISTS {name}")

    # 提交更改
    conn.commit()

    # 关闭游标和数据库连接
    cur.close()
    conn.close()
