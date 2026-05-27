from fastapi import FastAPI
import psycopg2 as psql
import dbInfo


connect = psql.connect(dbname = dbInfo.DB.dbname, user = dbInfo.DB.username, password =  dbInfo.DB.password, host = dbInfo.DB.ip, port = "5432")
app = FastAPI()


@app.get("/api/shops/{id}")
def get_shop(id:int):
   
    with connect.cursor() as cursor:
        cursor.execute(f"") #DB query goes here
        return cursor.fetchall()


