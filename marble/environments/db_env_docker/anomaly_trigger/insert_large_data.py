import datetime

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


"""insert_large_data"""


def insert_large_data(threads, duration, ncolumns, nrows, colsize, table_name="table1"):
    print_time()
    # Delete undeleted tables
    delete_table(table_name)
    # create a new table
    create_table(table_name, colsize, ncolumns)
    db = Database(init())
    # insert the data
    # insert_definitions = ', '.join(f'repeat(round(random()*999)::text,{(colsize//3)})' for i in range(ncolumns))
    insert_definitions = ", ".join(
        f"(SELECT substr(md5(random()::text), 1, {colsize}))" for i in range(ncolumns)
    )
    insert_data = f"insert into {table_name} select generate_series(1,{nrows}),{insert_definitions}, now();"
    db.concurrent_execute_sql(threads, duration, insert_data, commit_interval=1)

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
    num_columns = 10

    # Number of rows to insert
    num_rows = 100

    # Size of each column (in characters)
    column_size = 200

    # Table name
    table_name = "table1"

    # Call the insert_large_data function
    insert_large_data(
        num_threads, insert_duration, num_columns, num_rows, column_size, table_name
    )
