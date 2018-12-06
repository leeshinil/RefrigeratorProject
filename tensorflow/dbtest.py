import pymysql

conn = pymysql.connect(host='refrigerator.ckh68t0iet4w.us-east-2.rds.amazonaws.com',port=3306 ,user='lexshinil',password='12345678',db='refrigerator')
curs = conn.cursor()

sql = "select * from expired where name = 'onion'"

#name = ("onion")

#filename = ("2018-12-02 14:11:04")
#test = ("1234")
#val = (name, filename, test)
#print(val)

curs.execute(sql)
rows = curs.fetchall()

#sql2 = "insert into refrigerator (name, inputdate, expiration) values ('orange','123','123')"
#curs.execute(sql2)
#conn.commit()
print(rows)

conn.close()

