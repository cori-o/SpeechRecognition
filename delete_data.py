from src import DBConnection, TableEditor, PostgresDB
import os 
import json

with open(os.path.join('./config', 'db_config.json')) as f:
    db_config = json.load(f)

db_conn = DBConnection(db_config)
db_conn.connect()
postgres = PostgresDB(db_conn)
table_editor = TableEditor(db_conn)

conf_ids = postgres.get_mismatch_data()
for conf_id in conf_ids:
    table_editor.edit_poc_conf_log_tb(task='delete', table_name='ibk_poc_conf_log', data=conf_id)
    table_editor.edit_user_tb(task='delete', table_name='ibk_poc_conf_user', data=conf_id)
    table_editor.edit_poc_conf_tb(task='delete', table_name='ibk_poc_conf', data=conf_id)
