import os
import json


def create_directory_if_not_exists(folder_path):
    """
    检查文件夹路径是否存在，如果不存在则创建

    参数：
    - folder_path: 要检查和创建的文件夹路径
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def getFileList(directory):
    fileList = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            fileList.append(file_path)

        for subdir in dirs:
            fileList += getFileList(os.path.join(root, subdir))
    return fileList


"""
将字典对象保存为Json文件
"""


def save_json_file(path, item):
    item = json.dumps(item)

    if not os.path.exists(path):
        with open(path, "w", encoding='utf-8') as f:
            f.write(item + ",\n")
    else:
        with open(path, "a", encoding='utf-8') as f:
            f.write(item + ",\n")
