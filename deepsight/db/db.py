'''
# File: db.py
# Project: deepsight-back
# Created Date: 一月 8,2020 14:31
# Author: vito
# Description: 操作数据库增，改，查接口
# -----
# Last Modified: 2020/01/10 10:49
# Modified By: vito
# -----
# Copyright (C) DeepSight - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
'''

import json
import pymysql
import random
import time
import datetime
import logging
from django.conf import settings
from deepsight.util import snowflake
from deepsight import main

logger = logging.getLogger(settings.ENVTYPE)


class DbOperation:
    """
    DbOperation类打包操作数据库的方法，init初始化数据库各个字段默认值
    """
    def __init__(self, conn='', id='', sample_id=None, image_name=None, url=None, start_time=None,
                 finish_time=None, duration=None, is_completed=0, result=None, xml_url=None,
                 xml_downloaded=0, xml_name=None, create_time=None, update_time=None):
        self.conn = conn
        self.id = id
        self.sample_id = sample_id
        self.image_name = image_name
        self.url = url
        self.start_time = start_time
        self.finish_time = finish_time
        self.duration = duration
        self.is_completed = is_completed
        self.result = result
        self.xml_url = xml_url
        self.xml_downloaded = xml_downloaded
        self.xml_name = xml_name
        self.create_time = create_time
        self.update_time = update_time


    def db_insert(self):
        if self.conn.ping() is not None:
            logger.warning("Database connection failed, retry connection!")
            self.conn = main.DeepsightMain().db_connect()
        # 使用cursor()方法获取操作游标
        cur = self.conn.cursor()
        # 随机选取32位的uuid
        # self.id = ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',32))

        # 通过调用util工具包内的snowflake自增长id方式
        self.id = snowflake.get_snowflake()

        # 自动更新生成数据时间
        self.create_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 插入方法的sql语句
        sql_insert = """insert into %s(id,sample_id,image_name,url,start_time,finish_time,
                    duration,is_completed,result,xml_url,xml_downloaded,xml_name,create_time,update_time) """ % settings.DB_TABLE + """values(%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s)"""

        # 插入时查询sql语句
        sql_insert_select = """select * from %s """ % settings.DB_TABLE + """where url=%s;"""
        try:
            # 判断该插入字段是否在数据库存在
            cur.execute(sql_insert_select, (self.url))
            results = cur.fetchall()
            if results:
                pass
            else:
                try:
                    # 通过操作游标操作数据库，并引入类属性
                    cur.execute(sql_insert,
                                         (self.id, self.sample_id, self.image_name, self.url, self.start_time,
                                          self.finish_time, self.duration, self.is_completed, self.result,
                                          self.xml_url, self.xml_downloaded, self.xml_name, self.create_time,
                                          self.update_time))
                    # 提交操作数据库方法
                    self.conn.commit()
                    logger.info("insert 数据库完成！")
                except Exception as e:
                    # 检测到操作异常回滚操作
                    self.conn.rollback()
                    raise e
        except Exception as a:
            # 检测到操作异常日志记录
            logger.warning(a)
        finally:
            cur.close()



    def db_update_results(self):
        if self.conn.ping() is not None:
            logger.warning("Database connection failed, retry connection!")
            self.conn = main.DeepsightMain().db_connect()

        # 使用cursor()方法获取操作游标
        cur = self.conn.cursor()
        self.update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 更新数据库字段sql语句
        sql_update_results = """update %s """ %settings.DB_TABLE + """set start_time=%s,finish_time=%s,duration=%s,
        is_completed=%s,result=%s,update_time=%s where sample_id=%s and image_name=%s;"""
        try:
            cur.execute(sql_update_results, (self.start_time, self.finish_time, self.duration, self.is_completed,
                                             self.result, self.update_time, self.sample_id, self.image_name))
            self.conn.commit()
            logger.info("update 数据库完成！")
        except Exception as e:
            self.conn.rollback()
            logger.warning(e)
        finally:
            cur.close()
            # self.db.close()

    def db_update_xml(self):
        if self.conn.ping() is not None:
            logger.warning("Database connection failed, retry connection!")
            self.conn = main.DeepsightMain().db_connect()

        # 使用cursor()方法获取操作游标
        cur = self.conn.cursor()
        self.update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql_update_xml = """update %s """ %settings.DB_TABLE + """set xml_url=%s,xml_downloaded=%s,xml_name=%s,update_time=%s 
        where sample_id=%s and url=%s;"""
        try:
            cur.execute(sql_update_xml, (self.xml_url, self.xml_downloaded, self.xml_name,
                                         self.update_time, self.sample_id, self.url))
            self.conn.commit()
            logger.info("update 数据库完成！")
        except Exception as e:
            self.conn.rollback()
            logger.warning(e)
        finally:
            cur.close()
            # self.db.close()


    def db_select(self):
        if self.conn.ping() is not None:
            logger.warning("Database connection failed, retry connection!")
            self.conn = main.DeepsightMain().db_connect()
        # 使用cursor()方法获取操作游标
        cur = self.conn.cursor()
        # 查询数据库字段sql语句
        sql_select = """select image_name,url from %s """ %settings.DB_TABLE + """where sample_id=%s;"""
        try:
            cur.execute(sql_select, (self.sample_id))
            results = cur.fetchall()
            # print("id\tsample_id\timage_name\turl\tstart_time\tfinish_time\tduration\tis_completed\tresult\txml_url\txml_downloaded\txml_name\tcreate_time\tupdate_time")
            select_results = []
            for row in results:
                # id = row[0]
                # sample_id = row[1]
                # image_name = row[2]
                # url = row[3]
                # start_time = row[4]
                # finish_time = row[5]
                # duration = row[6]
                # is_completed = row[7]
                # result = row[8]
                # xml_url = row[9]
                # xml_downloaded = row[10]
                # xml_name = row[11]
                # create_time = row[12]
                # update_time = row[13]
                # print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(id, sample_id, image_name, url, start_time, finish_time, duration, is_completed, result, xml_url, xml_downloaded, xml_name, create_time, update_time))
                select_results.append(row)
            # 返回ai调取查询接口查询的数据库字段
            return select_results
        except Exception as e:
            logger.warning(e)
        finally:
            cur.close()
            # self.db.close()
