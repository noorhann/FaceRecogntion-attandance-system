import sqlite3
con = sqlite3.connect('dbtest.db')
cursor = con.cursor()

info=cursor.execute("SELECT * from instructors")
for row in info:
    print(row)
