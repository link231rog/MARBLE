import datetime
import os
import random
import time
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


# print the current time
def print_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_time)


"""missing_index"""


def missing_index(threads, duration, ncolumns, nrows, colsize, table_name="table1"):
    # create a new table
    print_time()
    db = Database(init())
    delete_table(table_name)
    create_table(table_name, colsize, ncolumns)

    # insert some data to be selected
    insert_definitions = ", ".join(
        f"(SELECT substr(md5(random()::text), 1, {colsize}))" for i in range(ncolumns)
    )
    insert_data = f"insert into {table_name} select generate_series(1,{nrows}),{insert_definitions}, now();"
    db.execute_sqls(insert_data)

    # select without the index
    missing_index = "select * from " + table_name + " where id="
    db.concurrent_execute_sql(threads, duration, missing_index, nrows)

    # delete the table
    delete_table(table_name)
    # print the end time
    print_time()


if __name__ == "__main__":
    # Number of threads to use for concurrent inserts
    num_threads = 100

    # Duration for which to run the inserts (in seconds)
    insert_duration = 60

    # Number of columns in the table
    num_columns = 75

    # Number of rows to insert
    num_rows = 37100

    # Size of each column (in characters)
    column_size = 100

    # Table name
    table_name = "table1"

    # Call the insert_large_data function

    missing_index(
        num_threads, insert_duration, num_columns, num_rows, column_size, table_name
    )
