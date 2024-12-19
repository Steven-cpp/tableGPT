from sqlalchemy import create_engine
import os

def connect_sql(env: str):
    server = os.getenv('SERVER_NAME')
    database = os.getenv('DATABASE_NAME')
    username = os.getenv('USER_ADMIN')
    password = os.getenv('PWD_ADMIN')
    driver = 'ODBC Driver 17 for SQL Server'

    if env == 'dev':
        server = server.replace('prd', 'dev')
        database = database.replace('prd', 'dev')

    connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver.replace(' ', '+')}"

    try:
        conn = create_engine(connection_string)
        print(f"[{env.upper()}] Connection to {database} successful!")
        return conn

    except Exception as e:
        print("Error:", e)
        return None