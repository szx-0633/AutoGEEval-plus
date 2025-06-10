import os
import shutil
import zipfile
import pandas as pd
import numpy as np
import rasterio

def safe_delete_directory(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))
        print(f"已删除 {path}，文件夹可能仍然存在，请自行删除空文件夹")


def split_and_compress_files(source_dir, target_dir, num_folders=20):
    """
    将source_dir中的文件平均分配到num_folders个文件夹，并对每个文件夹进行压缩。
    :param source_dir: 源文件夹路径，包含所有小文件
    :param target_dir: 目标文件夹路径，用于存放分组后的文件夹和压缩包
    :param num_folders: 分组的数量，默认为20
    """
    # 确保目标文件夹存在
    os.makedirs(target_dir, exist_ok=True)

    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    total_files = len(files)
    print(f"总文件数: {total_files}")

    # 计算每个文件夹应该包含的文件数量
    files_per_folder = total_files // num_folders
    remainder = total_files % num_folders

    # 创建文件夹并分配文件
    for i in range(num_folders):
        folder_name = os.path.join(target_dir, f"folder_{i+1}")
        os.makedirs(folder_name, exist_ok=True)

        # 计算当前文件夹应包含的文件范围
        start_idx = i * files_per_folder + min(i, remainder)
        end_idx = start_idx + files_per_folder + (1 if i < remainder else 0)

        # 将文件移动到对应的文件夹
        for file_name in files[start_idx:end_idx]:
            src_path = os.path.join(source_dir, file_name)
            dst_path = os.path.join(folder_name, file_name)
            shutil.move(src_path, dst_path)

        print(f"已将文件分配到 {folder_name}")

    # 压缩每个文件夹
    for i in range(num_folders):
        folder_name = os.path.join(target_dir, f"folder_{i+1}")
        zip_name = os.path.join(target_dir, f"part_{i+1}.zip")

        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files_in_folder in os.walk(folder_name):
                for file in files_in_folder:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, folder_name)
                    zipf.write(file_path, arcname)

        print(f"已压缩 {folder_name} 到 {zip_name}")

    # 可选：删除临时文件夹
    for i in range(num_folders):
        folder_name = os.path.join(target_dir, f"folder_{i+1}")
        shutil.rmtree(folder_name)
        print(f"已删除临时文件夹 {folder_name}")


# 合并包含算子组合信息的csv文件
def merge_csv_files(file1, file2, output_file):
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    df3 = df.drop_duplicates()
    df3.to_csv(output_file, index=False, encoding='utf-8', header=False)


# 将文件从latin1编码转换为utf-8编码
def reencode_file(input_file, output_file):
    with open(input_file, 'r', encoding='latin1') as infile:
        lines = infile.readlines()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(lines)

def clean_npy_txt_files(folder_path, config_path):
    """
    清理指定文件夹中的.npy.txt文件，并在配置文件中修改错误的输出文件名
    :param folder_path: 要清理的文件夹路径
    :param config_path: 配置文件路径，包含目标文件夹路径
    """
    with open(config_path, 'r') as f:
        lines = f.readlines()

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy.txt'):
            target_name = file_name.replace('.npy.txt', '.npy')
            for i, line in enumerate(lines):
                if target_name in line:
                    # 修改对应行的内容
                    lines[i] = line.replace('.npy', '.txt')

    with open(config_path, 'w') as f:
        f.writelines(lines)


def read_tiff(file_path):
    raster = rasterio.open(file_path).read()
    array = np.array(raster)
    array = np.round(array, 3)
    print("Array shape:", array.shape)
    print("Array pixels:\n", array)


if __name__ == "__main__":
    # safe_delete_directory('../gee_raw_script')
    split_and_compress_files('../ScriptAST/gee_raw_script_deduplicated', '../ScriptAST/gee_raw_script', num_folders=20)
    # clean_npy_txt_files('./test_code1/atomic_code/ref_answer', './test_code1/atomic_code/atomic_test_config.yaml')
    # merge_csv_files("./data/prefix_span/prefixspan_output_operators_0.2_6.csv",
    #                 "./data/custom_function_prefix_span/custom_functions_prefixspan_output_operators_0.05_6.csv",
    #                 "./data/prefix_span_merged_operators.csv")
    # read_tiff("D://Download/1.tiff")
    print(1)
    