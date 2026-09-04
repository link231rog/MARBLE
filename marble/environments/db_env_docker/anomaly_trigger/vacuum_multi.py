import datetime
import random
import time

from utils.database import DB_CONFIG, Database, DBArgs


def init():
    return DBArgs("postgresql", DB_CONFIG, application_name="anomaly")


# print the current time
def print_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(formatted_time)


"""vacuum"""


def vacuum(threads, duration, ncolumns, nrows, colsize, table_name="table1"):
    db = Database(init())
    # create a new table
    print_time()

    # insert some data to be deleted
    insert_definitions = ", ".join(
        f"(SELECT substr(md5(random()::text), 1, {colsize}))" for i in range(ncolumns)
    )
    insert_data = f"insert into {table_name} select generate_series(1,{nrows}),{insert_definitions}, now();"
    db.execute_sqls(insert_data)

    # delete 80% of the rows
    delete_nrows = int(nrows * 0.8)
    vacuum = f"delete from {table_name} where id < {delete_nrows};"
    db.execute_sqls(vacuum)

    # do the select , then the vacuum occurs
    select = "select * from " + table_name + " where id="
    db.concurrent_execute_sql(threads, duration, select, nrows)

    print_time()


if __name__ == "__main__":
    # Number of threads to use for concurrent inserts
    num_threads = 100

    # Duration for which to run the inserts (in seconds)
    insert_duration = 60

    # Number of columns in the table
    num_columns = 10

    # Number of rows to insert
    num_rows = 100

    # Size of each column (in characters)
    column_size = 20000

    # Table name
    table_name = "table1"

    # Call the insert_large_data function
    vacuum(num_threads, insert_duration, num_columns, num_rows, column_size, table_name)
