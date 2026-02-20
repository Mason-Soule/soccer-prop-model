import psycopg2

conn = psycopg2.connect(
    dbname="player_props",
    user="grownp",
    password="Mase3806",
    host="localhost",
    port="5432"
)

print("Connected successfully")
conn.close()
