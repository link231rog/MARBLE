import datetime
import random
import time
from multiprocessing.pool import Pool

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


"""redundent_index"""


def redundent_index(
    threads, duration, ncolumns, nrows, colsize, nindex, table_name="table1"
):
    # create a new table
    print_time()
    delete_table(table_name)
    create_table(table_name, colsize, ncolumns)
    db = Database(init())
    # insert some data to be updated
    insert_definitions = ", ".join(
        f"(SELECT substr(md5(random()::text), 1, {colsize}))" for i in range(ncolumns)
    )
    insert_data = f"insert into {table_name} select generate_series(1,{nrows}),{insert_definitions}, now();"
    db.execute_sqls(insert_data)

    # initialization of the indexes
    nindex = int((nindex * ncolumns) / 10)
    db.build_index(table_name, nindex)
    id_index = "CREATE INDEX index_" + table_name + "_id ON " + table_name + "(id);"
    db.execute_sqls(id_index)

    # lock_contention
    pool = Pool(threads)
    for _ in range(threads):
        pool.apply_async(lock, (table_name, ncolumns, colsize, duration, nrows))
    pool.close()
    pool.join()

    # drop the index
    db.drop_index(table_name)

    # delete the table
    delete_table(table_name)
    print_time()


def lock(table_name, ncolumns, colsize, duration, nrows):
    db = Database(init())
    start = time.time()
    # lock_contention
    while time.time() - start < duration:
        conn = db.resetConn()
        cur = conn.cursor()
        while time.time() - start < duration:
            col_name = random.randint(0, ncolumns - 1)
            row_name = random.randint(1, nrows - 1)
            lock_contention = f"update {table_name} set name{col_name}=(SELECT substr(md5(random()::text), 1, {colsize})) where id ={row_name}"
            # db.concurrent_execute_sql(threads,duration,lock_contention,nrows)
            cur.execute(lock_contention)
            conn.commit()
        conn.commit()
        conn.close()


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
    column_size = 200

    # Table name
    table_name = "table1"

    nindex = 6

    # Call the insert_large_data function
    redundent_index(
        num_threads,
        insert_duration,
        num_columns,
        num_rows,
        column_size,
        nindex,
        table_name,
    )
