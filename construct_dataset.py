from call_language_model import call_language_model
from prompts import CONSTRUCT_ATOMIC, CONSTRUCT_COMBINATION, CONSTRUCT_THEME_2, CONSTRUCT_THEME_NEW, CLARIFY_DOCSTRING
from typing import Optional, List, Tuple, Dict
import os
import json
import re
import pandas as pd
import ast
import shutil
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import yaml


# 原子需求部分
def generate_gee_atomic_test_code(operators_file, output_path, model_name="qwen-max-2025-01-25", model_provider="aliyun"):
    os.makedirs(output_path, exist_ok=True)

    operators_data = pd.read_excel(operators_file)
    operators_data = operators_data[["full_name","description","output_type","parameters"]]

    count = 0

    for index, row in operators_data.iterrows():
        # construct json information
        operator_name = row["full_name"]
        explanation = row["description"]
        return_type = row["output_type"]
        parameters = row["parameters"]

        # keywords = ['evaluate', 'aside', 'export', 'getinfo', 'map', 'load', 'create', 'constructor', "set", "blob",
        #              "url", "object", "processing", "random", "apply", "model", "authenticate", "call", "classifier",
        #              "clusterer", "ee.data", "iterate", "ee.join", "ee.kernel", "initialize", "ee.reducer", "ee.Filter",
        #              "ee.pixeltype", "ee.projection"]
        # operator_name_lower = operator_name.lower()
        # if any(keyword in operator_name_lower for keyword in keywords):
        #     print(f"Skipping {operator_name}")
        #     count += 1
        #     continue

        operator_json = {
            "operator_name": operator_name,
            "explanation": explanation,
            "return_type": return_type,
            "parameters": parameters
        }

        out_file_path = f"{output_path}/{operator_name}.txt"

        # Skip if the output already exists
        if os.path.exists(f"{output_path}/{operator_name}.txt"):
            print(f"Skipping {operator_name}")
            count += 1
            continue

        output, token_usage, error = call_language_model(
            model_provider=model_provider,
            model_name=model_name,
            system_prompt="You are an expert in geospatial analysis with python and Google Earth Engine(GEE).",
            user_prompt=CONSTRUCT_ATOMIC+json.dumps(operator_json),
            temperature=0.1,
            max_tokens=8192,
            config_path=r"./llm_config.yaml")

        with open(out_file_path, "w", encoding='utf-8') as f:
            f.write(output)

        print(f"Processed {operator_name}")
        print(f"Token usage: {token_usage}")
        count += 1
        print(count)
        # if count >= 1:
        #     break


def process_raw_file(raw_file_path, code_path, config_path):
    """
    处理原始 txt 文件，提取 Python 代码和 YAML 测试用例。

    :param raw_file_path: 原始 txt 文件路径
    :param code_path: 提取的 Python 代码保存路径
    :param config_path: 合并后的 YAML 文件保存路径，包含文件名
    """

    # 确保保存路径存在
    os.makedirs(code_path, exist_ok=True)
    # 初始化用于存储所有 YAML 内容的字符串
    all_yaml_content = ""
    count = 0
    function_names = set()  # 用于存储函数名，避免重复
    all_yaml_content = []
    # 遍历所有 txt 文件
    for file_name in os.listdir(raw_file_path):
        if not file_name.endswith('.txt'):
            continue  # 跳过非 txt 文件

        file_path = os.path.join(raw_file_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 提取 Python 代码块
        python_code_match = re.search(r'### Standard Code\s*```python\s*(.*?)\s*```', content, re.DOTALL)
        if python_code_match:
            python_code = python_code_match.group(1).strip()

            # 提取函数名
            function_name_match = re.search(r'def\s+(\w+)\(', python_code)
            if function_name_match:
                function_name = function_name_match.group(1)
                # 添加函数名到集合
                function_names.add(function_name)
                # 保存 Python 代码到单独文件
                code_file_path = os.path.join(code_path, f"{function_name}.txt")
                with open(code_file_path, 'w', encoding='utf-8') as code_file:
                    code_file.write(python_code)

        # 提取 YAML 测试用例
        yaml_match = re.search(r'### Test Cases\s*```yaml\s*(.*?)\s*```', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
            yaml_content_lines = yaml_content.splitlines()
            for line in yaml_content_lines:
                if not line.startswith(" ") and not line.startswith("-") and ":" in line:
                    line = f"{function_name}:"
                all_yaml_content.append(line)
            all_yaml_content += "\n"  # 将 YAML 内容追加到字符串中

        count += 1

    # 将所有 YAML 内容保存到文件
    if all_yaml_content:
        all_yaml_content = "\n".join(all_yaml_content)
        with open(config_path, 'w', encoding='utf-8') as config_file:
            config_file.write(all_yaml_content.strip())  # 写入合并后的 YAML 内容

    print(f"Processed {count} files")


# 组合需求部分
def convert_string_to_sequence(input_file, header=True, key='none'):
    if header:
        df = pd.read_csv(input_file)
    else:
        df = pd.read_csv(input_file, header=None)
    sequences = []

    for idx, row in df.iterrows():
        try:
            # 使用ast.literal_eval将字符串转换为Python列表
            sequence = ast.literal_eval(row[key])
            sequences.append(sequence)
        except (SyntaxError, ValueError) as e:
            print(f"错误处理第{idx}行: {e}")
            print(f"问题字符串: {row[key]}")
            sequences.append([])  # 添加空列表作为占位符

    return sequences, df


def process_single_operator(args: Tuple[List, int, str, str, str]) -> Tuple[bool, str]:
    """处理单个算子的函数"""
    operator_list, count, output_path, model_name, model_provider = args

    operators = str(operator_list)
    out_file_path = f"{output_path}/task{count + 1}.txt"

    try:
        # 如果输出已存在则跳过
        if os.path.exists(out_file_path):
            return True, f"Skipped {operator_list}"

        output, token_usage, error = call_language_model(
            model_provider=model_provider,
            model_name=model_name,
            system_prompt="You are an expert in geospatial analysis with python and Google Earth Engine(GEE).",
            user_prompt=CONSTRUCT_COMBINATION + operators,
            temperature=0.1,
            max_tokens=8192,
            config_path=r"./llm_config.yaml")

        header = f"""### Operators: {str(operator_list)}\n\n"""
        with open(out_file_path, "w", encoding='utf-8') as f:
            f.write(header)
            f.write(output)

        return True, f"Processed {operators} (Token usage: {token_usage})"
    except Exception as e:
        return False, f"Error processing {operators}: {str(e)}"


def generate_gee_combined_test_code(operator_lists, output_path, model_name="qwen-max-2025-01-25",
                                    model_provider="aliyun", max_workers=4):
    """并行版本的代码生成函数"""
    os.makedirs(output_path, exist_ok=True)

    # 准备参数列表
    args_list = [
        (operator_list, i, output_path, model_name, model_provider)
        for i, operator_list in enumerate(operator_lists)
    ]

    # 使用线程池执行并行任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_operator, args): args
            for args in args_list
        }

        # 使用tqdm创建进度条
        with tqdm(total=len(args_list), desc="Processing operators") as pbar:
            for future in concurrent.futures.as_completed(futures):
                args = futures[future]
                try:
                    success, message = future.result()
                    if success:
                        tqdm.write(message)  # 使用tqdm.write避免与进度条冲突
                    else:
                        tqdm.write(f"\033[91m{message}\033[0m")  # 红色显示错误
                except Exception as e:
                    tqdm.write(f"\033[91mError in task {args}: {str(e)}\033[0m")
                finally:
                    pbar.update(1)

def process_raw_file_combined(raw_file_path, code_path, config_path):
    """
    处理原始 txt 文件，提取 Python 代码和 YAML 测试用例。

    :param raw_file_path: 原始 txt 文件路径
    :param code_path: 提取的 Python 代码保存路径
    :param config_path: 合并后的 YAML 文件保存路径，包含文件名
    """

    # 确保保存路径存在
    os.makedirs(code_path, exist_ok=True)
    # 初始化用于存储所有 YAML 内容的字符串
    all_yaml_content = ""
    count = 0
    function_names = set()  # 用于存储函数名，避免重复
    all_yaml_content = []
    # 遍历所有 txt 文件
    for file_name in os.listdir(raw_file_path):
        if not file_name.endswith('.txt'):
            continue  # 跳过非 txt 文件

        file_path = os.path.join(raw_file_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 提取 Python 代码块
        python_code_match = re.search(r'### Standard Code\s*```python\s*(.*?)\s*```', content, re.DOTALL)
        if python_code_match:
            python_code = python_code_match.group(1).strip()

            # 提取函数名
            function_name_match = re.search(r'def\s+(\w+)\(', python_code)
            if function_name_match:
                function_name = function_name_match.group(1)
                # 检查函数名是否已存在
                function_name_new = function_name + "_" + str(count)
                python_code = python_code.replace(function_name, function_name_new)
                function_name = function_name_new

                # 添加函数名到集合
                function_names.add(function_name)

                # 保存 Python 代码到单独文件
                code_file_path = os.path.join(code_path, f"{function_name}.txt")
                with open(code_file_path, 'w', encoding='utf-8') as code_file:
                    code_file.write(python_code)

        # 提取 YAML 测试用例
        yaml_match = re.search(r'### Test Cases\s*```yaml\s*(.*?)\s*```', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
            yaml_content_lines = yaml_content.splitlines()
            for line in yaml_content_lines:
                if not line.startswith(" ") and not line.startswith("-") and ":" in line:
                    line = f"{function_name}:"
                all_yaml_content.append(line)
            all_yaml_content += "\n"  # 将 YAML 内容追加到字符串中

        count += 1

    # 将所有 YAML 内容保存到文件
    if all_yaml_content:
        all_yaml_content = "\n".join(all_yaml_content)
        with open(config_path, 'w', encoding='utf-8') as config_file:
            config_file.write(all_yaml_content.strip())  # 写入合并后的 YAML 内容

    print(f"Processed {count} files")


# 场景需求部分
def process_single_theme_task(task_data, output_path, task_id, model_name, model_provider):
    """处理单个任务的函数"""
    out_file_path = f"{output_path}/task{task_id}.txt"

    # 如果文件已存在，可以选择跳过
    if os.path.exists(out_file_path):
        return f"Task {task_id} already exists, skipped"

    instruction = task_data.get("instruction", "")
    input_text = task_data.get("input", "")
    output_text = task_data.get("output", "")

    # 移除output中的<think>...</think>部分
    output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL)
    role_string = "You are an expert in geospatial analyzing and coding. You are required to generate geospatial analysis code based on given code summary."
    instruction = instruction.replace(role_string, "")

    # 构建传递给模型的内容
    theme_content = f"""
    instruction: {instruction}

    input: {input_text}

    output: {output_text}
    """

    try:
        # 调用语言模型
        output, token_usage, error = call_language_model(
            model_provider=model_provider,
            model_name=model_name,
            system_prompt="You are an expert in geospatial analysis with python and Google Earth Engine(GEE).",
            user_prompt=CONSTRUCT_THEME_NEW + theme_content,
            temperature=0.1,
            max_tokens=8192,
            config_path=r"./llm_config.yaml")

        if error:
            return f"Error processing task {task_id}: {error}"

        # 写入输出文件
        header = f"""### Task: {task_id}\n\n"""
        with open(out_file_path, "w", encoding='utf-8') as f:
            f.write(header)
            f.write(output)

        return f"Processed task {task_id}, token usage: {token_usage}"

    except Exception as e:
        return f"Exception in task {task_id}: {str(e)}"


def generate_gee_theme_test(jsonl_file_path, output_path, model_name="qwen-max-2025-01-25",
                                model_provider="aliyun", max_workers=8, rate_limit_delay=1.0):
    """
    为jsonl文件中的每一行数据生成GEE测试代码和测试用例，使用多线程并行处理

    Args:
        jsonl_file_path: jsonl文件路径
        output_path: 输出目录路径
        model_name: 使用的模型名称
        model_provider: 模型提供商
        max_workers: 最大线程数
        rate_limit_delay: API调用间隔时间(秒)，避免过快请求导致限流
    """
    os.makedirs(output_path, exist_ok=True)

    # 读取所有任务数据
    tasks = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))

    print(f"Found {len(tasks)} tasks in the jsonl file")

    # 创建线程池执行任务
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # 提交所有任务
        for i, task_data in enumerate(tasks, 1):
            # 添加延迟以避免API限流
            if i > 1:
                time.sleep(rate_limit_delay)

            future = executor.submit(
                process_single_theme_task,
                task_data,
                output_path,
                i,
                model_name,
                model_provider
            )
            futures[future] = i

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tasks"):
            task_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(result)
            except Exception as e:
                print(f"Task {task_id} generated an exception: {e}")

    return results


def process_raw_file_new(raw_file_path, code_path, config_path):
    """
    处理原始 txt 文件，提取 Python 代码和 YAML 测试用例。

    :param raw_file_path: 原始 txt 文件路径
    :param code_path: 提取的 Python 代码保存路径
    :param config_path: 合并后的 YAML 文件保存路径，包含文件名
    """

    # 确保保存路径存在
    os.makedirs(code_path, exist_ok=True)
    # 初始化用于存储所有 YAML 内容的字符串
    all_yaml_content = ""
    count = 0
    function_names = set()  # 用于存储函数名，避免重复
    # 遍历所有 txt 文件
    for file_name in os.listdir(raw_file_path):
        if not file_name.endswith('.txt'):
            continue  # 跳过非 txt 文件

        file_path = os.path.join(raw_file_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        if "users" in content:
            continue

        # 提取 Python 代码块
        python_code_match = re.search(r'### Standard Code\s*```python\s*(.*?)\s*```', content, re.DOTALL)
        if python_code_match:
            python_code = python_code_match.group(1).strip()

            # 提取函数名
            function_name_match = re.search(r'def\s+(\w+)\(', python_code)
            if function_name_match:
                function_name = function_name_match.group(1)
                # 检查函数名是否已存在
                if function_name in function_names:
                    print(f"Warning: Function name '{function_name}' already exists. Skipping. File: {file_name}")
                    continue
                # 添加函数名到集合
                function_names.add(function_name)

                # 保存 Python 代码到单独文件
                code_file_path = os.path.join(code_path, f"{function_name}.txt")
                with open(code_file_path, 'w', encoding='utf-8') as code_file:
                    code_file.write(python_code)

        # 提取 YAML 测试用例
        yaml_match = re.search(r'### Test Cases in YAML Format\s*```yaml\s*(.*?)\s*```', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
            all_yaml_content += yaml_content + "\n\n"  # 将 YAML 内容追加到字符串中

        count += 1

    # 将所有 YAML 内容保存到文件
    if all_yaml_content:
        with open(config_path, 'w', encoding='utf-8') as config_file:
            config_file.write(all_yaml_content.strip())  # 写入合并后的 YAML 内容

    print(f"Processed {count} files")



def save_standard_tests(code_directory, instruction_directory, standard_code_directory):
    """
    从代码文件中提取到第二个三引号为止的内容，作为测试任务保存到新文件中。随后将原始文件移动到标准代码目录。
    假设函数定义从文件第一行开始。

    参数:
    code_directory (str): 包含Python函数文件的目录路径
    instruction_directory (str): 输出测试任务的目录路径
    standard_code_directory (str): 标准代码的目录路径

    返回:
    dict: 包含提取信息的字典
    """
    # 确保输出目录存在
    if not os.path.exists(instruction_directory):
        os.makedirs(instruction_directory)

    # 获取所有代码文件
    files = os.listdir(code_directory)

    # 用于存储提取信息的字典
    extraction_info = {
        'total_files': len(files),
        'extracted_tests': 0,
        'errors': 0,
        'details': []
    }

    for file in files:
        if not file.endswith('.txt'):
            continue

        file_path = os.path.join(code_directory, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_lines = f.readlines()

            # 计数三引号出现次数
            triple_quote_count = 0
            end_line = 0

            for i, line in enumerate(code_lines):
                # 计算当前行中三引号的出现次数
                quotes_in_line = line.count('"""')

                if quotes_in_line > 0:
                    # 如果一行中有多个三引号
                    if quotes_in_line > 1:
                        # 一行内有多个三引号，需要全部计数
                        triple_quote_count += quotes_in_line
                    else:
                        # 只有一个三引号的正常情况
                        triple_quote_count += 1

                    # 检查是否达到了第二个三引号
                    if triple_quote_count >= 2:
                        end_line = i
                        break

            # 如果找到了第二个三引号
            if triple_quote_count >= 2:
                # 提取从第一行到第二个三引号的内容
                test_content = ''.join(code_lines[0:end_line + 1])

                # 保存到输出文件
                test_name = file.replace('.txt', '')
                output_file = os.path.join(instruction_directory, f"{test_name}_test_instruction.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(test_content)

                extraction_info['extracted_tests'] += 1
                extraction_info['details'].append({
                    'file': file,
                    'output_file': output_file,
                    'lines_extracted': end_line + 1
                })
            else:
                # 没有找到第二个三引号
                print("未找到第二个三引号, 文件: ", file)
                extraction_info['errors'] += 1
                extraction_info['details'].append({
                    'file': file,
                    'error': "未找到第二个三引号"
                })

            # 移动原始文件到标准代码目录
            new_path = os.path.join(standard_code_directory, file)
            shutil.move(file_path, new_path)

        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            extraction_info['errors'] += 1
            extraction_info['details'].append({
                'file': file,
                'error': str(e)
            })


def extract_and_match_functions(code_directory, yaml_path, output_yaml_path):
    """
    从指定目录中提取Python函数名，在YAML配置中找到对应的测试用例，
    并将匹配的配置保存到新的YAML文件中。

    通过文本处理方式解析YAML，保留原始格式和标签。

    参数:
    code_directory (str): 包含Python函数文件的目录路径
    yaml_path (str): 原始YAML配置文件路径
    output_yaml_path (str): 输出的新YAML文件路径

    返回:
    bool: 操作是否成功
    """
    try:
        # 获取所有代码文件
        files = [f for f in os.listdir(code_directory) if os.path.isfile(os.path.join(code_directory, f))]

        # 从文件中提取函数名
        function_names = set()
        function_name_pattern = r"def (\w+)\("

        for file in files:
            file_path = os.path.join(code_directory, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                    matches = re.findall(function_name_pattern, code_content)
                    function_names.update(matches)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

        print(f"从代码文件中提取到的函数名: {function_names}")

        # 读取YAML文件的所有行
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_lines = f.readlines()

        # 解析YAML文件，将其分割成函数块
        function_blocks = {}
        current_block = []
        current_function = None

        for line in yaml_lines:
            # 检查是否是一个新的顶级条目（函数定义）
            # 顶级条目是不以空格开头且包含冒号的行
            if line.strip() and not line.startswith(' ') and not line.startswith('-') and ':' in line:
                # 如果已经有当前函数，保存它的块
                if current_function is not None and current_block:
                    function_blocks[current_function] = current_block

                # 提取新函数名（冒号前的部分）
                current_function = line.split(':', 1)[0].strip()
                current_block = [line]  # 开始一个新块，包含当前行
            elif current_function is not None:
                # 将行添加到当前块
                current_block.append(line)

        # 保存最后一个函数块
        if current_function is not None and current_block:
            function_blocks[current_function] = current_block

        # 创建新的YAML内容，只包含匹配的函数
        matched_blocks = []
        for func_name, block in function_blocks.items():
            if func_name in function_names:
                matched_blocks.extend(block)

        # 保存匹配的块到输出文件
        with open(output_yaml_path, 'w', encoding='utf-8') as f:
            f.writelines(matched_blocks)

        matched_count = sum(1 for func in function_blocks if func in function_names)
        print(f"成功匹配了 {matched_count} 个函数的配置，已保存到 {output_yaml_path}")
        return True

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_edge_test_cases(code_directory: str, yaml_path: str, output_path: str,
        model_provider: str = "aliyun", model_name: str = "qwen-max-2025-01-25", temperature: float = 0.7) -> None:
    """
    Generate edge test cases for GEE functions using LLM.

    Args:
        code_directory: Directory containing the code files
        yaml_path: Path to the original YAML config file
        output_path: Path to save the updated YAML file
        model_provider: LLM provider name
        model_name: Model name
        temperature: Temperature for LLM generation
    """

    def unknown_tag_handler(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            value = loader.construct_scalar(node)
            return f"!{tag_suffix} {value}"
        return None

    class CustomLoader(yaml.SafeLoader):
        pass

    yaml.add_multi_constructor('', unknown_tag_handler, CustomLoader)

    # Read the code files
    code_content = ""
    for file in os.listdir(code_directory):
        if file.endswith('.txt'):
            with open(os.path.join(code_directory, file), 'r', encoding='utf-8') as f:
                code_content += f.read() + "\n\n"

    # Read the YAML config as text
    with open(yaml_path, 'r', encoding='utf-8') as f:
        original_yaml_lines = f.readlines()
        f.close()

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=CustomLoader)
        f.close()

    # Create the prompt
    system_prompt = """You are an expert in Google Earth Engine (GEE) Python API programming and testing."""
    new_yaml_content = ""

    # Process each function
    count = 0
    for func_name, test_cases in config.items():
        if count < 700:
            count += 1
            continue

        print("Starting to process function:", func_name)

        config_content = ""
        in_config = False
        for line in original_yaml_lines:
            if line.startswith(str(func_name)):
                config_content += line
                in_config = True
            elif in_config:
                config_content += line
                if not line.startswith(' '):
                    in_config = False
            else:
                continue

        # Find the corresponding function definition in code
        code_file_path = os.path.join(code_directory, f"{func_name}.txt")
        if os.path.exists(code_file_path):
            with open(code_file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        else:
            print("Function code not found for:", func_name)
            continue

        user_prompt = f"""
        You are tasked with creating YAML-formatted test cases for a Python function that wraps a Google Earth Engine (GEE) algorithm.
        Your are required to add **1–2** edge test cases (with edge_test: true) to the existing set of test cases, without altering their structure or indentation.
        
        ### Input
        1. You are given the function signature, docstring, and executable code.
        2. You are also given the existing test cases in YAML format.
        Each test case includes parameters (params), where GEE special objects(e.g. ee.Image) is defined using a ! python tag that returns a GEE object.
        Scalar parameters like sigma and threshold are also included.
        Each test case also contains:
        expected_answer: A filename (.npy for arrays/images, .geojson for geometry and features, and .txt for other types).
        out_type: Expected output type (e.g., ee.Image, bool).
        edge_test: Boolean indicating whether this is an edge/boundary test case (true or false).
        
        ### Edge case examples
        These include, but are not limited to:
        1. Invalid geometry (e.g., empty geometry, coordinates out of bounds)
        2. Empty images (e.g., from empty collections or masked-out regions)
        3. Non-existent assets (e.g., invalid image IDs)
        4. Invalid dates (e.g., malformed date strings)
        5. Array/list index out of bounds or empty lists
        6. Invalid parameter types or values
        7. Missing necessary arguments
        8. Numerical overflow or division by zero
        9. Invalid projections or coordinate reference systems (CRS)
        
        ### Requirements
        1. Add 1–2 new edge test cases below the existing ones. If already 2 edge test cases exist, do NOT add more.
        2. Do NOT modify the structure or indentation of existing test cases. Do NOT change or output the function code.
        
        The function code:
            {code_content}
        The existing test cases:
            {config_content}
        Please provide the new test cases in YAML format, ensuring that the indentation and structure are consistent with the existing ones.
        Return ONLY the YAML content (including original test cases), NO additional text or explanation.
        """

        # Call the language model
        response, tokens, error_msg = call_language_model(
            model_provider=model_provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )

        if error_msg:
            print(f"Error calling LLM for {func_name}: {error_msg}")
            continue

        # Extract pure YAML content
        yaml_content = response
        if "```yaml" in response:
            yaml_content = response.split("```yaml")[1].split("```")[0].strip()
        elif "```" in response:
            yaml_content = response.split("```")[1].strip()

        if yaml_content:
            # Add the function name as a key
            yaml_content = yaml_content.replace("### Test Cases", "")
            yaml_content = yaml_content.replace("```yaml", "")
            yaml_content = yaml_content.replace("```", "")
            yaml_content = yaml_content.strip()

            # Append the new YAML content to the output
            new_yaml_content += yaml_content
            new_yaml_content += "\n\n"
        else:
            new_yaml_content += config_content
            print(f"No YAML content generated for {func_name}!")

        count += 1
        print("Processed number:", count, "Tokens", tokens)
        if count%100 == 0:
            temp_path = output_path.replace(".yaml", f"_temp{count}.yaml")
            with open(temp_path.replace(".yaml", f""), 'w', encoding='utf-8') as f:
                f.write(new_yaml_content)
                f.close()

        # if count >= 1:
        #     break

    # Save the new YAML content
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_yaml_content)


def generate_gee_theme_test_2(raw_file_path, output_path, model_name="qwen-max-2025-01-25",
                                model_provider="aliyun", max_workers=4, rate_limit_delay=1.0):
    os.makedirs(output_path, exist_ok=True)

    # 读取所有任务数据
    tasks = []
    files = os.listdir(raw_file_path)
    for file_name in files:
        with open(os.path.join(raw_file_path, file_name), 'r', encoding='utf-8') as f:
            content = f.read()
            tasks.append(content)

    print(f"Found {len(tasks)} tasks")

    # 创建线程池执行任务
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # 提交所有任务
        for i, task_data in enumerate(tasks, 1):
            # 添加延迟以避免API限流
            if i > 1:
                time.sleep(rate_limit_delay)

            future = executor.submit(
                process_single_theme_task_2,
                task_data,
                output_path,
                i,
                model_name,
                model_provider
            )
            futures[future] = i

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing tasks"):
            task_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(result)
            except Exception as e:
                print(f"Task {task_id} generated an exception: {e}")

    return results


def process_single_theme_task_2(task_data, output_path, task_id, model_name, model_provider):
    """处理单个任务的函数"""
    out_file_path = f"{output_path}/task{task_id}.txt"

    # 如果文件已存在，可以选择跳过
    if os.path.exists(out_file_path):
        return f"Task {task_id} already exists, skipped"

    raw_code = task_data

    try:
        # 调用语言模型
        output, token_usage, error = call_language_model(
            model_provider=model_provider,
            model_name=model_name,
            system_prompt="You are an expert in geospatial analysis with python and Google Earth Engine(GEE).",
            user_prompt=CONSTRUCT_THEME_2 + raw_code,
            temperature=0.1,
            max_tokens=8192,
            config_path=r"./llm_config.yaml")

        if error:
            return f"Error processing task {task_id}: {error}"

        # 写入输出文件
        header = f"""### Task: {task_id}\n\n"""
        with open(out_file_path, "w", encoding='utf-8') as f:
            f.write(header)
            f.write(output)

        return f"Processed task {task_id}, token usage: {token_usage}"

    except Exception as e:
        return f"Exception in task {task_id}: {str(e)}"


def process_clarify_file(args):
    file_path, output_path, model_provider, model_name, temperature = args
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code_content = f.read()

        system_prompt = """You are an expert in Google Earth Engine (GEE) Python API."""
        user_prompt = CLARIFY_DOCSTRING + code_content

        # 调用语言模型
        response, tokens, error_msg = call_language_model(
            model_provider=model_provider,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
        if error_msg:
            return False, error_msg, 0

        response = response.replace("```python", "").replace("```", "").strip()
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(response)

        return True, tokens, None

    except Exception as e:
        return False, str(e), 0


def clarify_docstring(code_directory: str, output_directory: str, model_provider: str = "aliyun",
                      model_name: str = "qwen-max-2025-01-25", temperature: float = 0.3, max_workers: int = 10):
    os.makedirs(output_directory, exist_ok=True)

    tasks = []
    for file in os.listdir(code_directory):
        if file.endswith('.txt'):
            input_path = os.path.join(code_directory, file)
            output_path = os.path.join(output_directory, file)
            tasks.append((input_path, output_path, model_provider, model_name, temperature))

    count = 0
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_clarify_file, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            success, data, _ = future.result()
            if success:
                count += 1
                total_tokens += data
            else:
                print(f"Error: {data}")

    print(f"Successfully processed {count} files. Total tokens used: {total_tokens}")



if __name__== "__main__":

    # 1. 原子需求部分
    # 生成原子需求的原始数据
    # operators_file = "./data/GEE_API_Atomic.xlsx"
    # output_path = "./test_code/autogen_atomic_raw"
    # model_name = "qwen-max-2025-01-25"
    # model_provider = "aliyun"
    # generate_gee_atomic_test_code(operators_file, output_path, model_provider=model_provider, model_name=model_name)
    # 处理原始数据，提取Python代码和YAML测试用例
    # raw_file_path = "./test_code/autogen_atomic_raw"
    # code_path = "./test_code/atomic_code"
    # config_path = "./test_code/atomic_code/atomic_test_config.yaml"
    # process_raw_file(raw_file_path, code_path, config_path)
    # 提取测试任务
    # code_directory = "./test_code/atomic_code"
    # instruction_directory = "./test_code/atomic_code/test_instructions"
    # standard_code_directory = "./test_code/atomic_code/standard_code"
    # save_standard_tests(code_directory, instruction_directory, standard_code_directory)
    # # 提取函数名并匹配YAML配置
    # code_path = "./test_code/atomic_code/standard_code"
    # yaml_path = "./test_code/atomic_code/atomic_test_config.yaml"
    # output_yaml_path = "./test_code/atomic_code/matched_atomic_test_config.yaml"
    # extract_and_match_functions(code_path, yaml_path, output_yaml_path)
    # # 生成边界测试用例
    # code_directory = "./test_code/atomic_code/standard_code"
    # yaml_path = "./test_code/atomic_code/atomic_test_config.yaml"
    # output_yaml_path = "./test_code/atomic_code/atomic_test_config_edge.yaml"
    # model_provider = "aliyun"
    # model_name = "qwen-max-2025-01-25"
    # generate_edge_test_cases(code_directory, yaml_path, output_yaml_path,
    #                          model_provider=model_provider, model_name=model_name)

    # 2. 组合需求部分
    # 生成组合需求的原始数据，请运行frequent_pattern_mining.py脚本
    # 生成组合需求的原始数据
    # operators_file = "./data/combined_operators_list_0.1_0.04_150.csv"
    # output_path = "./test_code/autogen_combined_raw"
    # model_name = "qwen-max-2025-01-25"
    # model_provider = "aliyun"
    # operators_list, _ = convert_string_to_sequence(operators_file, key="processed")
    # generate_gee_combined_test_code(operators_list, output_path, model_provider=model_provider, model_name=model_name)
    # 处理原始数据，提取Python代码和YAML测试用例
    # raw_file_path = "./test_code/autogen_combined_raw"
    # code_path = "./test_code/combined_code"
    # config_path = "./test_code/combined_code/combined_test_config.yaml"
    # process_raw_file_combined(raw_file_path, code_path, config_path)
    # 提取测试任务
    code_directory = "./test_code/combined_code"
    # instruction_directory = "./test_code/combined_code/test_instructions"
    # standard_code_directory = "./test_code/combined_code/standard_code"
    # save_standard_tests(code_directory, instruction_directory, standard_code_directory)
    # 清晰化注释
    clarified_combined_directory = "./test_code/combined_code/clarified_code"
    clarify_docstring(code_directory, clarified_combined_directory, model_provider="aliyun",
                      model_name="qwen-max-2025-01-25", temperature=0.3)


    # 3. 场景需求部分
    # raw_file_path = "./test_code/autogen_theme_raw"
    # code_path = "./test_code/theme_code"
    # config_path = "./test_code/theme_code/theme_test_config.yaml"
    # model_name = "qwen-max-2025-01-25"
    # model_provider = "aliyun"
    # jsonl_file_path = "./data/geocode_alpaca_1k.jsonl"
    # generate_gee_theme_test(jsonl_file_path, raw_file_path, model_name, model_provider)
    # process_raw_file_new(raw_file_path, code_path, config_path)
    # 提取测试任务
    # code_directory = "./test_code/theme_code"
    # instruction_directory = "./test_code/theme_code/test_instructions"
    # standard_code_directory = "./test_code/theme_code/standard_code"
    # save_standard_tests(code_directory, instruction_directory, standard_code_directory)

    print(1)


    