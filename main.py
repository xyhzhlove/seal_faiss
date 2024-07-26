#coding=utf-8
from datetime import datetime
import faiss
import numpy as np
import os
import pickle
# from model_config import *
import pymysql
from pydantic import BaseModel, Field
from typing import List
import json


# 测试数据
import requests
from docs import ppt_docs,pdf_docs

# 一次可以将多少个文本转为向量
vector_batch = 32
# 向量服务地址
vector_url = "http://192.168.2.171:30080"
def text_embedding(data):
    data = {
       "inputs": data
    }

    headers = {
       "Content-Type": "application/json"
    }

    response = requests.post(vector_url, headers=headers, json=data)

    if response.status_code == 200:
      response_data = response.json()
      return response_data
    #  这里返回的是向量的一维列表，一个文本转之后的向量就是一个一维列表;该服务返回的向量是1024维度的
    else:
       # print(f"请求失败，状态码: {response.status_code}")
       return None



"""
mysql封装
"""
"""
faiss添加向量 操作的是 二维列表的numpy化，即[[指定维度的向量1],[指定维度的向量2],[指定维度的向量3]]
faiss查询向量 操作的也是 二维列表的numpy化，即[[需要查询的向量]]
"""





class DocumentFormat(BaseModel):
    """Interface for interacting with a document."""
    complete_content: str = ""
    sentence: str = ""
    is_title: int = 1
    is_head: int = 1
    level: int = 0
    first_directory: str = ""
    second_directory: str = ""
    outline: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
class MysqlHelper:
    # todo 数据库连接参数，可以定义多个，比如conn_params1，conn_params2，用于连接多个数据库，在类实例化时指定


    # todo 类的构造函数，主要用于类的初始化
    def __init__(self, conn_params):
        self.__host = conn_params['host']
        self.__port = conn_params['port']
        self.__db = conn_params['db']
        self.__user = conn_params['user']
        self.__passwd = conn_params['password']
        self.__charset = conn_params['charset']
        self.__connect()


    # todo 建立数据库连接和打开游标
    def __connect(self):
        self.__conn = pymysql.connect(host=self.__host,
                              port=self.__port,
                              db=self.__db,
                              user=self.__user,
                              password=self.__passwd,
                              charset=self.__charset)
        # 游标初始化
        self.__cursor = self.__conn.cursor()

    # todo 关闭游标和关闭连接
    def __close(self):
        self.__cursor.close()
        self.__conn.close()

    """
    描述:插入一条数据
    data:为所插入数据的字典形式
    table_name为表名
    callback:数据库在插入后执行的向量数据库操作。当向量数据库操作成功后数据库插入事务才提交。出异常时数据库插入事务就回滚
    """
    def insert_one_data(self,data,table_name,callback):
        try:

            # 获取此时mysql的最大插入id值
            max_id_sql = f"SELECT MAX(id) as max_id FROM {table_name}"
            # 执行查询最大ID值的操作
            self.__cursor.execute(max_id_sql)
            max_id_result = self.__cursor.fetchone()
            # mysql开始插入的id
            start_id = max_id_result['max_id'] if max_id_result else None


            # 准备 SQL 插入语句
            sql = "INSERT INTO your_table_name (column1, column2) VALUES (%s, %s)"
            # 执行 SQL 语句
            self.__cursor.execute(sql, ('value1', 'value2'))  # 传入需要插入的值

            # mysql插入结束后的id
            last_id = self.__cursor.lastrowid

            callback(start_id,last_id)

            # 提交事务
            self.__conn.commit()
        except Exception as e:
            print(e)
            # 发生错误时回滚
            self.__conn.rollback()


    """
    描述，插入多条数据
    data_list: 为插入的数据列表
    table_name:表命
    callback:数据库在插入后执行的向量数据库操作。当向量数据库操作成功后数据库插入事务才提交。出异常时数据库插入事务就回滚
    """

    def insert_many_data(self,data_list,table_name,callback):
        try:
            # 获取此时mysql的最大插入id值
            max_id_sql = f"SELECT MAX(id) as max_id FROM {table_name}"
            # 执行查询最大ID值的操作
            self.__cursor.execute(max_id_sql)
            max_id_result = self.__cursor.fetchone()
            # mysql开始插入的id
            # print(max_id_result)
            # start_id = max_id_result['max_id'] if max_id_result else None


            # 准备 SQL 插入语句
            sql = f"INSERT INTO {table_name} (complete_content,sentence,is_title,is_head,level,first_directory,second_directory,metadata,outline,time) VALUES (%s, %s,%s,%s,%s,%s,%s,%s,%s,%s)"
            # 准备要插入的数据列表
            # 执行 SQL 语句
            self.__cursor.executemany(sql, data_list)

            # mysql插入数据的起始id
            start_id = self.__cursor.lastrowid

            callback(start_id)

        # 提交事务
            self.__conn.commit()
        except Exception as e:
            print(e)
            # 发生错误时回滚
            self.__conn.rollback()


    """
    描述:删除一条数据
    table_name: 表名称(空间名称)
    id_value: 删除的条件键名
    callback: 数据库在删除后执行的向量数据库操作。当向量数据库操作成功后数据库插入事务才提交。出异常时数据库插入事务就回滚
    """
    def remove_one_data(self,table_name, id_value,callback):
        try:
            sql = f"DELETE FROM {table_name} WHERE id = %s"
            self.__cursor.execute(sql, (id_value,))
            callback()
            self.__conn.commit()
            print(f"Row with id {id_value} deleted from {table_name}.")
        except Exception as e:
            print(f"MySQL Error: {e}")
            self.__conn.rollback()

    def delete_table_by_table_name(self,table_name,callback):

            try:
                # 执行删除表的语句
                sql = f"DROP TABLE IF EXISTS {table_name};"
                self.__cursor.execute(sql)
                callback()
                # 提交事务
                self.__conn.commit()
            except Exception as e:
                print(e)
                self.__conn.rollback()


    def remove_many_data(self,table_name,id_values,callback):
        try:
            # 对于每个主键值，生成一个删除语句
            if len(id_values) < 1:
                raise Exception('不存在符合条件的删除数据')
                return
            placeholders = ', '.join(['%s'] * len(id_values))
            combined_sql = f"DELETE FROM {table_name} WHERE id IN ({placeholders})"
            # 执行合并后的SQL命令
            self.__cursor.execute(combined_sql, id_values)
            #                       执行删除动作
            callback()
            # callback执行不报异常时才会真的将数据删除
            self.__conn.commit()
            # print(f"{len(id_values)} rows deleted from {table_name}.")
        except Exception as e:
            print(f"MySQL Error: {e}")
            # callback报异常时将数据回滚到删除前的状态
            self.__conn.rollback()


    """
    查询一条数据的方法
    id:查询数据的id
    """
    def search_one_data(self,table_name,id):
        self.__cursor.execute(f"SELECT * FROM {table_name} WHERE id = %s", (id))
        # 获取查询结果
        return self.__cursor.fetchone()

    """
    根据ids列表查询多条数据的方法
    """
    def search_many_data_by_ids(self,table_name,ids):
        # ids = (1, 2, 3, 4, 5)
        self.__cursor.execute(f"SELECT * FROM {table_name} WHERE id IN (%s)", (ids))

        # 获取所有查询结果
        return self.__cursor.fetchall()


    """
    根据文件名查询多条数据
    file_name:文件名称,table_name:空间名称
    """
    def search_many_data_ids_by_file_name(self,file_name,table_name):

        sql = f"SELECT * FROM {table_name} WHERE JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.source')) = %s"
        name_to_find = file_name

        # 执行查询
        self.__cursor.execute(sql, (name_to_find,))

        # 获取查询结果
        results = self.__cursor.fetchall()
        # print(results)

        ids = [item[0] for item in results]
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        return ids
        # print(ids)

    def search_many_data_ids_by_file_list(self,file_list,table_name):

        placeholders = ', '.join(['%s'] * len(file_list))
        sql = f"SELECT * FROM {table_name} WHERE JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.source')) IN ({placeholders})"
        # 执行查询
        self.__cursor.execute(sql, file_list)

        # 获取查询结果
        results = self.__cursor.fetchall()
        # print(results)

        ids = [item[0] for item in results]
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        return ids
        # print(ids)



    def search_many_data_content_by_id_list(self,id_list,table_name):
        # print(id_list)
        # print(table_name)
        if len(id_list) < 1:
            raise  Exception('没有查到相似的结果')
            return
        placeholders = ','.join(['%s'] * len(id_list))
        sql = f"SELECT * FROM {table_name} WHERE id IN ({placeholders})"
        # print(sql)

        # 执行查询
        self.__cursor.execute(sql, id_list)

        # 获取查询结果
        results = self.__cursor.fetchall()


        content_list = [item[2] for item in results]
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        return content_list
        # print(ids)

    """
    描述:创建数据库表
    """
    def table_is_exists(self,table_name):
            # 创建游标
            check_table_sql = f"SHOW TABLES LIKE '{table_name}';"
            self.__cursor.execute(check_table_sql)
            tables = self.__cursor.fetchall()
            # 判断表是否存在
            if not tables:
                # 表不存在
                return False
            else:
                # 表存在
                return True
    def create_table(self,table_name,callback):
        try:
            create_table_sql = f"""CREATE TABLE IF NOT EXISTS {table_name}(
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  complete_content TEXT,
                  sentence VARCHAR(256),
                  is_title INT,
                  is_head INT,
                  level INT,
                  first_directory VARCHAR(255),
                  second_directory VARCHAR(255),
                  outline JSON,
                  metadata JSON,
                  time DATETIME)
            """
            # print(create_table_sql)
            self.__cursor.execute(create_table_sql)
            callback()
            # callback函数执行成功不出异常时，创建表格的动作(事务)才会真正完成(提交)
            self.__conn.commit()  # 提交事务
        except Exception as e:
            print(e)
            # callback函数执行失败出异常时，创建表格的动作(事务)不会成功(提交)，会回滚到创建表格之前的状态
            self.__conn.rollback()  # 提交事务

"""
内容入库时先入mysql的库，通过mysql的id再将Id作为自定义id入向量faiss库
"""

"""
向量数据库封装
"""
class MyFaiss:
    """
    collections_path: 空间的存储的磁盘路径
    dimension: 向量维度
    """
    def __init__(self,collections_path,dimension:int=1024):
        self.collections_path = collections_path
        self.dimension = dimension
        # 空间与faiss文件的映射
        self.space_map = self.get_collections()
#         当前选择的空间/索引的实例(index实例)
        self.current_index = None
        self.embeddings = None
        # print(self.space_map)
        self.current_space_name = None
        if not self.collections_path:
            raise Exception('请传入空间索引映射关系的存储文件')


        conn_params = {
            'host': 'localhost',
            'port': 3306,
            'user': 'root',
            'password': '123456',
            'db': 'zsk',
            'charset': 'utf8'}
        #mysql实例
        self.sqlInstance = MysqlHelper(conn_params)



    # 创建空间
    def create_collection(self, vs_name:str):
        """描述: 创建空间
        vs_name: 空间名称
        """

        def create_faiss():    # 创建空间
            if vs_name in self.space_map:
                raise Exception(f'{vs_name}空间已经存在')
            else:
                index = faiss.IndexFlatIP(self.dimension)
                index = faiss.IndexIDMap(index)
                index_path = self.save_index(index,vs_name)
                self.space_map[vs_name] = index_path
            # 存储/更新空间缓存
                self.save_collections()

        self.sqlInstance.create_table(vs_name,create_faiss)



    """
    return_value : 缓存的 空间:索引路径字典
    """
    def get_collections(self):
        # 判断空间存储路径是否存在
        if os.path.exists(self.collections_path):
            data = {}
            with open(self.collections_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            data = {}
            with open(self.collections_path, 'wb') as file:
                pickle.dump(data, file)
            return data




    """
    将空间与索引路径的映射字典映写到本地的缓存文件中
    """
    def save_collections(self):
        # 序列化并保存索引对象到文件，存储空间
        with open(self.collections_path, 'wb') as f:
            pickle.dump(self.space_map, f)
        print(f'更新{self.collections_path}信息')

    """
    collect_name: 空间名称
    描述: 设置当前空间
    self.current_index = self.read_index(collect_name)
    return_value: None
    """
    def set_collection(self, collect_name):
        if collect_name in self.space_map:
                self.current_index = self.read_index(collect_name)
                self.current_space_name = collect_name
        else:
            raise Exception('空间不存在')



    """
    index: faiss创建的索引对象
    collection_name: 空间名
    return_value: faiss文件路径
    描述:将索引写入到索引文件中，返回索引路径
    """
    def save_index(self,index,collection_name):
        if not os.path.exists('indexes'):
            os.makedirs('indexes')

        faiss_path =  f"indexes\{collection_name}.faiss"
        faiss.write_index(index, faiss_path)
        print(f'将{collection_name}最新的向量信息写入对应的faiss文件')
        return faiss_path

    """
    collection_name: 集合名称
    return_value: 返回faiss的索引对象
    描述:读取索引文件
    """
    def read_index(self,collection_name):
        index_path = f"indexes\{collection_name}.faiss"
        if not os.path.exists(index_path):
            raise Exception('所操作空间不存在')
        index = faiss.read_index(index_path)
        print(f'从{collection_name}faiss文件中读出当前的索引信息')
        return index


    """
    collection_name: 空间名称
    描述:创建索引
    """
    def create_index(self, collection_name):
        try:
           if collection_name in self.space_map:
               if self.space_map[collection_name] is None:
                   index = faiss.IndexFlatIP(self.dimension)
                   index = faiss.IndexIDMap(index)
                   index_path = self.save_index(index, collection_name)
                   self.space_map[collection_name] = index_path
           else:
               raise Exception('空间不存在,请先创建空间')
        except Exception as e:
            print(e)


    """
    描述:加载空间
    vs_name: 空间名称
    embedding: 向量模型实例
    
    """
    def load_collection(self, vs_name, embedding):
        try:
          self.set_collection(vs_name)
          self.embeddings = embedding
        except Exception as e:
            print(e)


    """
    描述:检查空间是否存在
    collection_name: 空间名称
    return_value:False | True
    """
    def check_collection_exist(self, collection_name):
        has = False
        if collection_name in self.space_map:
            if self.space_map.get(collection_name):
                 has = True
        return has





    def delete_milvus_table(self, collection_name):
        """
            完成
            描述:删除milvus表格
            collect_name: 空间名称
            需要做的:需要找到该空间的所有向量Id，然后先去mysql表格中删除向量Id,保证mysql删除成功后再删除空间对应的faiss文件，并更新空间缓存文件(self.space_map在磁盘上的文件)
            """
        #
        # self.set_collection(collection_name)
        # 获取空间的id去mysql中删除多项；然后删除空间对应的faiss文件，如果删除空间faiss文件成功，那么就提交删除数据库的事务
        ids = self.read_json2ids(collection_name)

        def remove_space_faiss_file():
            faiss_file_path = f'indexes\{collection_name}.faiss'
            ids_file_path = f'ids\{collection_name}_ids.json'
            if os.path.exists(faiss_file_path):
               os.remove(faiss_file_path)
            if os.path.exists(ids_file_path):
               os.remove(ids_file_path)
            # print(self.space_map)
            # 将该空间从空间管理中移除
            del self.space_map[collection_name]
            # print(self.space_map)
            # 重新记录空间管理中的空间信息
            self.save_collections()
        self.sqlInstance.delete_table_by_table_name(collection_name,remove_space_faiss_file)


    def query_by_file_list(self, collection_name, file_name_list):
          """
           完成
            collection_name: 空间名称
            file_name_list:文件名称列表
            return_value: 返回所需文件在mysql中的所有id，根据文件名查出在mysql中的id，返回id列表。
            """
          try:
              ids = self.sqlInstance.search_many_data_ids_by_file_list(file_name_list,collection_name)
              return ids
              # print(ids)
          except Exception as e:
              print(e)

    def query_by_file(self, collection_name, file_name):
            """
            完成
            collection_name: 空间名称
            file_name: 文件名称
            描述:去mysql中查询该文件名，并把文件名包含的所有id号返回
            """
            try:
                ids = self.sqlInstance.search_many_data_ids_by_file_name(file_name,collection_name)
                # print(ids)
                return ids
            except Exception as e:
                print(e)




    def delete_document_milvus(self, collection_name, file_name):

            """
            完成

            collection_name: 空间名称
            file_name: 文件名称
            先去mysql中根据文件名查询到符合的向量id,然后使用一个事务去删除mysql的id，然后删除索引文件中的id即可，不过mysql中的事务提交时一定得索引删除id之后再提交
            """
    #         去mysql中查询指定文件名的id号
    #           然后先去faiss(读出这些向量)文件中删除这些id的向量并更新Ids文件，
                # 将删除向量的索引重新写入文件中

            indices_to_delete = []
            # 需要删除的id列表
            try:
               indices_to_delete = self.sqlInstance.search_many_data_ids_by_file_name(file_name,collection_name)
            except Exception as e:
                print(e)
                return

            def remove_faiss():
                # print(1)
                # 获取当前索引文件中有的所有自定义向量索引
                ids = self.read_json2ids(collection_name)
                # 获取删除一些指定向量索引后的自定义向量索引
                remain_ids = [item for item in ids if item not in indices_to_delete]
                # 设置当前操作空间
                self.set_collection(collection_name)
                # 将删除的索引列表转为一维的numpy列表
                indices_to_delete_array = np.array(indices_to_delete, dtype='int64')

                # print(self.current_index.ntotal)
                # 从源码中得知的方法
                self.current_index.remove_ids(indices_to_delete_array)
                # 删除指定向量索引后剩余的向量数量
                # print(self.current_index.ntotal)
                # 将最新的索引重新写入文件中
                self.save_index(self.current_index,collection_name)
                # 并且更新当前索引对应的ids文件
                self.save_ids2json(collection_name,remain_ids)
            self.sqlInstance.remove_many_data(collection_name,indices_to_delete,remove_faiss)


    def save_ids2json(self,vs_name,ids):
        """
           描述:保存一个空间中的所有id，将id保存为一个json文件
           vs_name: 空间名称
           ids:该次写入的id
           """
        if not os.path.exists('ids'):
            os.makedirs('ids')
        file_path = f'ids\{vs_name}_ids.json'
        with open(file_path, 'w',encoding='utf-8') as f:
            json.dump(ids, f)
        print(f'{vs_name}_ids.json文件更新完毕')


    def read_json2ids(self,vs_name):
        file_path = f'ids\{vs_name}_ids.json'
        if os.path.exists(file_path):
            ids = []
            with open(file_path, 'r',encoding='utf-8') as f:
               ids = json.load(f)
            print(f'{vs_name}_ids文件读取')
            return ids
        else:
            ids = []
            if not os.path.exists('ids'):
                os.makedirs('ids')
            with open(file_path, 'w',encoding='utf-8') as f:
                json.dump(ids, f)
            print(f'{vs_name}_ids文件读取')
            return ids


    def add_document(self, docs:List[DocumentFormat], vs_name):


        """
            完成
            class DocumentFormat(BaseModel):
                Interface for interacting with a document
                complete_content: str# 完整的内容
                content: str# 256字内的内容
                title: str# 标题
                head: int# 是否为开头，开头为0，不是开头为1
                level: int # 几级目录
                first_directory: str# 一级目录
                second_directory: str# 二级目录
                metadata: dict = Field(default_factory=dict),
                outline: dict

            docs：用于录入文档片段的列表；示例[DocumentFormat()]
            vs_name: 空间名称
            描述: 遍历docs列表，将docs列表的每一项使用事务放入mysql中，得到每一项的Id，得到每一项的Id后，将这些项的文本向量化，然后根据这些id，自定义Id写入索引中
            等到完全写入成功后，再提交mysql事务
            """
        if self.check_collection_exist(vs_name):
            # self.load_collection(vs_name,1)
            self.set_collection(vs_name)
        else:
            raise Exception('空间不存在')
            # self.create_collection(vs_name)
            # self.load_collection(vs_name,1)

        def faiss_insert(start_id):
            # print(start_id,last_id)
           end_id = start_id + len(docs) - 1
           doc_sentences = [item.sentence for item in docs]
           # 每间隔chunk_size个将一个列表分为多个列表
           def split_list(lst, chunk_size):
                return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

           doc_sentences = split_list(doc_sentences,vector_batch)
           # print(doc_sentences)
           vectors = []
           for item_list in doc_sentences:
               item_vector_list = self.embeddings(item_list)
               for item_vector in item_vector_list:
                    vectors.append(item_vector)
           vectors = np.array(vectors)
           ids = [item for item in range(start_id, end_id + 1)]
            # 添加带有自定义索引号的向量
           self.current_index.add_with_ids(vectors, ids)



            # 生产环境时变为真实的向量添加
            # vectors = self.embeddings.encode(batch_sentences, normalize_embeddings=True)
           # vectors = []

           # self.current_index.add_with_ids(vectors, ids)


            # 检查是否正确添加
           # print("Total number of vectors in the index:", self.current_index.ntotal)
           # print("Index ids:", self.current_index.id_map.keys())



            # 保存索引到文件
           self.save_index(self.current_index,vs_name)

            # 保存该空间的索引中的自定义id信息
           #  先读再存
           origin_ids = self.read_json2ids(vs_name)
           current_ids = origin_ids + ids
           # print(current_ids)

           self.save_ids2json(vs_name,current_ids)

           # print("Index saved to index_with_ids.faiss")

           # query_vectors = np.random.random((2, self.dimension)).astype('float32')
           # D, I = self.current_index.search(query_vectors, 3)  # 查找最近的 3 个邻居
           #
           # print("距离:\n", D)
           # print("索引:\n", I)
           #
           #  距离:
           #  [[66.67971 65.13687 64.97006]
           #   [71.52849 71.52843 69.67891]]
           #  索引:
           #  [[140 146 163]
           #   [148 150 156]]

        if not self.sqlInstance.table_is_exists(vs_name):
            self.sqlInstance.create_table(vs_name)
        data_list = []
        for item in docs:
            complete_content = item.complete_content
            sentence = item.sentence
            is_title = item.is_title
            is_head = item.is_head
            level = item.level
            first_directory = item.first_directory
            second_directory = item.second_directory
            metadata = json.dumps(item.metadata)
            outline = json.dumps(item.outline)
            now = datetime.now()
            # 将日期和时间格式化为MySQL 'DATETIME' 类型所期望的字符串格式
            current_time = now.strftime('%Y-%m-%d %H:%M:%S')

            data_item = (complete_content,sentence,is_title,is_head,level,first_directory,second_directory,metadata,outline,current_time)
            data_list.append(data_item)
        self.sqlInstance.insert_many_data(data_list,vs_name,faiss_insert)

    def get_context_milvus(self, ids_list, result_list):
     """
     ids_list是每个向量的id的列表集合，是与该问题最相似的几个向量的Id的列表
     result_list: 数据库拉取到的相关结果(上下文数量是每个id的上下id查找范围是CONTEXT_NUM决定的)
     将这些结果按照相似的id号分别重组上下文
     最终返回上下文列表"""
     # print(result_list)
     try:
         CONTEXT_NUM = 5
         context_num = CONTEXT_NUM
         for index,id in enumerate(ids_list):
            add_id_list = [id + var for var in range(1,context_num+1)]
            add_result_list = self.sqlInstance.search_many_data_content_by_id_list(add_id_list,self.current_space_name)
            for res in add_result_list:
                 result_list[index] += f'{res}'
         # print(result_list)
         return result_list
     except Exception as e:
         print(e)

     # pass



    def similarity_IP_search(self, query, limit_num=3):
        """
        qyery: 用户提问的问题(文本形式)
        limit_num: 找相似度的前几个最相似的
        描述:先将问题向量化，然后使用faiss的相似度对比查找出相似的向量，找出相似度最高的前limit_num向量的id作为id_list，然后
        根据向量id去mysql中查，然后返回相似的文本结果，放在一起作为result_list
        """
        # 设置当前空间

        # 拿到向量服务后的真实数据
        # 将输入的文本向量化
        # vectors_to_search = [self.embeddings.encode(query, normalize_embeddings=True)]
        # distances, indices = self.current_index.search(vectors_to_search, limit_num)
        # # 相似度最高的前几个向量对应的索引
        # id_list = indices[0]
        # # 相似度最高的前几个向量对应的距离
        # distance_list = distances[0]

        try:
        # 模拟数据
            query_vectors = text_embedding(query)
            query_vectors = np.array(query_vectors)
            # print(query_vectors)

            D, I = self.current_index.search(query_vectors, limit_num)  # 查找最近的 3 个邻居

            # 将numpy数组转为列表

            # 找到相似的向量id的列表
            id_list = I[0].tolist()
            # print(type(id_list))
            # 找到相似的向量id对应的内容的列表
            result_list = self.sqlInstance.search_many_data_content_by_id_list(id_list,self.current_space_name)

            return self.get_context_milvus(id_list,result_list)
            # print(type(id_list))
            # print(result_list)
        except Exception as e:
            print(e)


