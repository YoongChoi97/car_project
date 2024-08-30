
from ftplib import FTP
import os

def get_all_jpg_files(directory):
    """지정된 디렉토리에서 모든 .jpg 파일 경로를 가져옵니다."""
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]

def send_images(file_paths, server_address=('localhost', 1004)):
    # FTP 연결
    ftp = FTP()
    print("FTP 서버에 연결 중...")
    ftp.connect(server_address[0], server_address[1])
    print("FTP 서버에 연결되었습니다.")
    ftp.login('add', '1004')

    # 여러 파일 전송
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"파일 업로드 중: {file_name}")
        try:
            with open(file_path, 'rb') as file:
                ftp.storbinary(f'STOR {file_name}', file)
            print(f"파일 업로드가 완료되었습니다: {file_name}")
        except Exception as e:
            print(f"파일 업로드 중 오류 발생: {file_name}, 오류: {e}")

    # FTP 연결 종료
    ftp.quit()
    print("FTP 연결이 종료되었습니다.")

if __name__ == '__main__':
    directory = './crops'  # 이미지 파일이 있는 디렉토리 경로
    file_paths = get_all_jpg_files(directory)
    send_images(file_paths)
