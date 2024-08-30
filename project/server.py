import os
import sqlite3
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer
from datetime import datetime

SERVER_HOST = 'localhost'
SERVER_PORT = 1004
SAVE_DIR = './car_save_image'
DATABASE_FILE = './car.db'

FTP_USER = 'add'
FTP_PASSWORD = '1004'


class CarDB:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()

    def create_table(self, table_name):
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                hour INTEGER,
                car_plate TEXT,
                file_name TEXT,
                line TEXT,
                upload_time TEXT
            )
        ''')
        self.conn.commit()

    def insert_data(self, table_name, year, month, day, hour, car_plate, file_name, line, upload_time):
        self.cursor.execute(f'''
            INSERT INTO {table_name} (year, month, day, hour, car_plate, file_name, line, upload_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (year, month, day, hour, car_plate, file_name, line, upload_time))
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()


def parse_file_name(file_name):
    try:
        file_name1 = file_name.split('-')
        # 날짜와 시간 부분 파싱
        year = int(file_name1[0][:4])
        month = int(file_name1[0][4:6])
        day = int(file_name1[0][6:8])
        hour = int(file_name1[0][8:10])

        # # 파일 이름의 나머지 부분을 '-'로 분할
        # parts = file_name[11:].split('-')
        # if len(parts) < 2:
        #     raise ValueError("Filename format is incorrect")

        car_plate = file_name1[1]
        line = file_name1[2][0]# 라인 정보는 마지막 하이픈 이후 첫 번째 문자로 가정

        if not car_plate or car_plate.lower() == "none":
            raise ValueError("Car plate is missing or invalid")

        return year, month, day, hour, car_plate, line
    except Exception as e:
        print(f"파일 이름 파싱 오류: {e}")
        return None


def save_file_metadata(file_name):
    file_info = parse_file_name(file_name)
    if not file_info:
        print("파일 이름 파싱 실패. 데이터베이스에 저장되지 않음.")
        return

    year, month, day, hour, car_plate, line = file_info
    db = CarDB(DATABASE_FILE)
    db.create_table('car_Table')
    db.insert_data('car_Table', year, month, day, hour, car_plate, file_name, line, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    db.close()
    print("데이터베이스에 성공적으로 저장되었습니다.")


class CustomFTPHandler(FTPHandler):
    def on_file_received(self, file):
        save_file_metadata(os.path.basename(file))


def run_ftp_server():
    authorizer = DummyAuthorizer()
    authorizer.add_user(FTP_USER, FTP_PASSWORD, SAVE_DIR, perm='elradfmw')

    handler = CustomFTPHandler
    handler.authorizer = authorizer

    db = CarDB(DATABASE_FILE)
    db.create_table('car_Table')
    db.close()

    server = FTPServer((SERVER_HOST, SERVER_PORT), handler)
    print(f'FTP 서버가 {SERVER_HOST}:{SERVER_PORT}에서 실행 중입니다.')
    server.serve_forever()


if __name__ == '__main__':
    run_ftp_server()

