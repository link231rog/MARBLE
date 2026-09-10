import subprocess

import createdatabase
import dropdatabase
from utils.database import DB_CONFIG, Database, DBArgs


def init():
    return DBArgs("postgresql", DB_CONFIG, application_name="anomaly")


# create a table
def create_table(table_name, colsize, ncolumns):
    db = Database(init())
    column_definitions = ", ".join(
        f"name{i} varchar({colsize})" for i in range(ncolumns)
    )
    creat_sql = (
        f"CREATE TABLE {table_name} (id int, {column_definitions}, time timestamp);"
    )
    db.execute_sqls(creat_sql)


# delete the table
def delete_table(table_name):
    db = Database(init())
    delete_sql = f"DROP TABLE if exists {table_name}"
    db.execute_sqls(delete_sql)


def run_command(command):
    """运行给定的命令"""
    process = subprocess.Popen(command, shell=True)
    return process


def main():
    # 定义要运行的命令
    command1 = "python anomaly_trigger/miss_multi.py"
    command2 = "python anomaly_trigger/insert_multi.py"
    dropdatabase.dropdatabase("tmp")
    createdatabase.createdatabase("tmp")
    # 同时启动两个进程
    table_name = "table1"
    delete_table(table_name)
    create_table(table_name, 1000, 1000)
    process1 = run_command(command1)
    process2 = run_command(command2)

    # 等待进程完成
    process1.wait()
    process2.wait()
    print("Both processes have finished.")
    dropdatabase.dropdatabase("tmp")


if __name__ == "__main__":
    main()
