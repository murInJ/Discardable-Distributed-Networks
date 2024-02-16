import os

def create_directory_if_not_exists(folder_path):
    """
    检查文件夹路径是否存在，如果不存在则创建

    参数：
    - folder_path: 要检查和创建的文件夹路径
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
