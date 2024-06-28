import os

def find_wav_files(folder_path, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    f.write(file_path + '\n')

# 设置文件夹路径和输出文件名
folder_path = '/home/wl/qkc/dataset/test'
output_file = '/home/wl/qkc/dataset/test_list_100.txt'

# 调用函数查找.wav文件并将路径写入到txt文件中
find_wav_files(folder_path, output_file)
