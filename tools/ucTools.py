import psycopg2
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from itertools import product
import os

class ucTools:

    def __init__(self, ADDRESS='172.16.25.20', PORT='5432', USERNAME='uqzshi3', PASSWORD='uqkingstdio2019', DBNAME='ec_dataset'):
        self.POSTGRES_ADDRESS = ADDRESS
        self.POSTGRES_PORT = PORT
        self.POSTGRES_USERNAME = USERNAME
        self.POSTGRES_PASSWORD = PASSWORD
        self.POSTGRES_DBNAME = DBNAME

    def db_conn(self):
        '''
        连接数据库方法
        '''
        # A long string that contains the necessary Postgres login information
        postgres_str = (
            'postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.
            format(username=self.POSTGRES_USERNAME,
                   password=self.POSTGRES_PASSWORD,
                   ipaddress=self.POSTGRES_ADDRESS,
                   port=self.POSTGRES_PORT,
                   dbname=self.POSTGRES_DBNAME))
        # Create the connection
        cnx = create_engine(postgres_str, poolclass=NullPool)
        return cnx

    # disease列表存入数据库
    def saveToDB(self, df_data, table_name, engine):
        try:
            df_data.to_sql(table_name, engine, index = False, if_exists ='append')
        except Exception as e:
            print(e)

    # update Table
    def update(self, sql):
        '''
        更新数据表
        '''
        with self.db_conn().begin() as conn:
            conn.execute(sql)

    def get_graph_dic(self):
        '''
        生成节点字典
        '''
        dic = {}
        j = 0
        for i in product([0, 1, 2, 3, 4], 6):
            j = j + 1
            dic[i]  = '%d' % (j)
        return dic