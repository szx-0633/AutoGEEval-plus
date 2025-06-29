import os
import json
import csv
import pandas as pd
import ast

from spmf import Spmf
from utils import merge_csv_files
import re
import esprima
import zipfile
import tempfile
import shutil
from tqdm import tqdm
import glob
from typing import Set, List, Optional, Dict, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# Define a function to extract all operators from AST
def extract_operators_from_ast_py(script_data, skip_custom_function=False, custom_function_tag=None):
    """
    Extract operator sequences from Python AST (GEE API calls).

    Args:
        script_data: raw script data (string)
        skip_custom_function: Whether to skip operators in custom functions
        custom_function_tag: Add special tag for operators in custom functions, e.g. "[FUNCTION]"

    Returns:
        List of extracted operator sequences
    """
    operators = []
    processed_nodes = set()  # Track processed nodes
    function_stack = []  # Function scope stack
    processed_functions = set()  # Track processed function definitions

    def is_in_function_scope(path):
        """Check if current node is inside a function and return the nearest unprocessed FunctionDef"""
        for node in reversed(path):
            if isinstance(node, ast.FunctionDef) and id(node) not in processed_functions:
                return node
        return None

    def get_function_name(func_node):
        """Get name of function definition"""
        return func_node.name if isinstance(func_node, ast.FunctionDef) else 'anonymous'

    def process_call_expression(node, in_function):
        """Process CallExpression-like node in Python AST"""
        node_id = id(node)
        if node_id in processed_nodes:
            return
        processed_nodes.add(node_id)

        operator = extract_operator_path_py(node.func)
        if operator:
            if in_function and custom_function_tag:
                operators.append(f"{custom_function_tag}{operator}")
            else:
                operators.append(operator)

    def recurse(node, path=None):
        if path is None:
            path = []

        if not isinstance(node, (ast.AST, list, tuple)):
            return

        current_path = path + [node] if isinstance(node, ast.AST) else path

        if isinstance(node, ast.FunctionDef):
            func_id = id(node)
            if func_id not in processed_functions:
                processed_functions.add(func_id)
                function_name = get_function_name(node)
                function_stack.append(function_name)
                if custom_function_tag:
                    operators.append("[FUNCTION_START]")

                # Process body
                for stmt in node.body:
                    recurse(stmt, current_path)

                if custom_function_tag:
                    operators.append("[FUNCTION_END]")
                function_stack.pop()
            return

        elif isinstance(node, ast.Call):
            in_function = bool(function_stack)
            if not skip_custom_function or not in_function:
                process_call_expression(node, in_function)

        # Recurse into child nodes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    recurse(item, current_path)
            elif isinstance(value, ast.AST):
                recurse(value, current_path)

    ast_obj = ast.parse(script_data)
    recurse(ast_obj)
    return operators


def extract_operator_path_py(node):
    """
    Extract operator path from a Call.func node in Python AST.
    For example, image.select(...).updateMask(...) -> select.updateMask
    """
    parts = []
    current = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)

    return '.'.join(reversed(parts)) if parts else None


def extract_operators_from_ast_js(ast_data, skip_custom_function=False, custom_function_tag=None):
    """
    Extract operator sequences from AST
    Args:
        ast_data: AST data
        skip_custom_function: Whether to skip operators in custom functions, default is False
        custom_function_tag: Add special tag for operators in custom functions, such as [function], default is None
    Returns:
        operators: List of extracted operator sequences
    """
    operators = []
    processed_nodes = set()  # Used to record processed nodes
    function_stack = []  # Used to track function nesting
    processed_functions = set()  # Used to record processed function declarations

    def is_in_function_scope(path):
        """Check if the current path is within a function scope, and return the nearest unprocessed function declaration node"""
        for node in reversed(path):  # Check from inside to outside
            if (isinstance(node, dict) and
                    node.get('type') == 'FunctionDeclaration' and
                    id(node) not in processed_functions):
                return node
        return None

    def get_function_name(func_node):
        """Get function name from function declaration node"""
        if func_node and 'id' in func_node and func_node['id'].get('name'):
            return func_node['id']['name']
        return 'anonymous'

    def process_call_expression(node, in_function):
        """Process CallExpression node, ensure correct execution order"""
        if not isinstance(node, dict) or node.get('type') != 'CallExpression':
            return

        node_id = id(node)
        if node_id in processed_nodes:
            return
        processed_nodes.add(node_id)

        callee = node.get('callee')
        if not callee or callee.get('type') != 'MemberExpression':
            return

        if callee['object'].get('type') == 'CallExpression':
            process_call_expression(callee['object'], in_function)

        operator = extract_operator_path_js(callee)
        if operator:
            if in_function and custom_function_tag:
                operators.append(f"{custom_function_tag}{operator}")
            else:
                operators.append(operator)

    def recurse(node, path=[]):
        if not isinstance(node, (dict, list)):
            return

        current_path = path + [node] if isinstance(node, dict) else path

        if isinstance(node, dict):
            # Check if entering a new function scope
            func_node = is_in_function_scope(current_path)

            if func_node:
                func_id = id(func_node)
                if func_id not in processed_functions:
                    processed_functions.add(func_id)
                    function_name = get_function_name(func_node)
                    function_stack.append(function_name)
                    if custom_function_tag:
                        operators.append(f"[FUNCTION_START]")

                    # Process function body
                    if 'body' in func_node:
                        recurse(func_node['body'], current_path)

                    # Function end
                    if custom_function_tag:
                        operators.append(f"[FUNCTION_END]")
                    function_stack.pop()
                    return  # Already processed function body, no need to continue

            # Process current node
            if node.get('type') == 'CallExpression' and (not skip_custom_function or not function_stack):
                process_call_expression(node, bool(function_stack))

            # Recursively process child nodes
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    recurse(value, current_path)

        else:  # isinstance(node, list)
            for item in node:
                recurse(item, current_path)

    recurse(ast_data)
    return operators


def extract_operator_path_js(callee):
    """Extract operator path"""
    operator_path = []
    current = callee

    # Process MemberExpression
    while current and current.get('type') == 'MemberExpression':
        if 'name' in current['property']:
            operator_path.append(current['property']['name'])
        current = current['object']

    # Process the final Identifier
    if current and current.get('type') == 'Identifier':
        operator_path.append(current['name'])

    return '.'.join(reversed(operator_path)) if operator_path else None


# Define a function to process all AST files in a folder
def process_single_zip_ast(zip_path, output_dir, part_num):
    """Process AST data in a single zip file"""
    # Create a CSV file for each part
    output_file = os.path.join(output_dir, f'operators_sequence_part_{part_num}.csv')
    processed_count = 0

    # Create a temporary directory for extracting files
    with tempfile.TemporaryDirectory(dir="./temp") as temp_dir:
        # Extract files to temporary directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Process extracted files
        files = [f for f in os.listdir(temp_dir) if f.endswith('.txt')]

        # Write directly to CSV, avoid storing in memory
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['file_name', 'operators'])  # Write header

            for filename in tqdm(files, desc=f"Processing {os.path.basename(zip_path)}"):
                file_path = os.path.join(temp_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        ast_data = json.load(f)
                        operators = extract_operators_from_ast_py(ast_data, skip_custom_function=False, custom_function_tag="[function]")
                        file_name = os.path.splitext(filename)[0]
                        # Convert operator list to string
                        operators_str = ','.join(operators)
                        writer.writerow([file_name, operators_str])
                        processed_count += 1
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

    return processed_count


def process_all_zips(zip_folder, output_dir, output_file):
    """Process all zip files"""
    os.makedirs(output_dir, exist_ok=True)

    total_processed = 0
    zip_files = sorted([f for f in os.listdir(zip_folder) if f.startswith('part_') and f.endswith('_ast.zip')])

    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        part_num = int(zip_file.split('_')[1])  # Extract part number from filename

        print(f"\nProcessing {zip_file}...")
        try:
            processed_count = process_single_zip_ast(zip_path, output_dir, part_num)
            total_processed += processed_count
            print(f"Completed {zip_file}: processed {processed_count} files")
        except Exception as e:
            print(f"Error processing {zip_file}: {str(e)}")
            continue
    print(f"\nTotal processed files: {total_processed}")


def merge_results(output_dir, output_file):
    """Merge all CSV result files into one final file"""
    # Match all files from operators_sequence_part_1.csv to operators_sequence_part_20.csv
    file_pattern = os.path.join(output_dir, 'operators_sequence_part_[1-9].csv')
    file_pattern_10_plus = os.path.join(output_dir, 'operators_sequence_part_1[0-9].csv')

    # Use glob to get the list of matching files
    files_1_9 = glob.glob(file_pattern)
    files_10_20 = glob.glob(file_pattern_10_plus)

    # Combine all file paths
    file_list = sorted(files_1_9 + files_10_20, key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Initialize an empty DataFrame to store merged data
    combined_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Remove rows where operators column is empty
    cleaned_df = combined_df.dropna(subset=['operators'])

    # Save results to output file
    cleaned_df.to_csv(output_file, index=False)

    print(f"Final results saved to: {output_file}")


def process_ast_folder(folder_path):
    all_operators = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                ast_data = json.load(f)
                operators = extract_operators_from_ast_py(ast_data)
                # Only keep filename (without path and extension) and operator list
                file_name = os.path.splitext(filename)[0]
                all_operators.append([file_name, operators])
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} files")
    print(f"Processed {count} files")

    return all_operators


def save_operators_to_csv(operators_list, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header, column names are File_Name and Operators
        writer.writerow(["File_Name", "Operators"])

        # Write filename and operator sequence (as a list) for each file
        for file_name, operators in operators_list:
            writer.writerow([file_name, str(operators)])


def remove_duplicates(input_csv_file, output_csv_file):
    seen_operators = set()  # Used to store processed operators
    unique_rows = []  # Store deduplicated rows

    with open(input_csv_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read header

        unique_rows.append(header)  # Add header to results

        for row in reader:
            operators = row[1].strip('[]')  # Remove square brackets from the list
            if operators not in seen_operators:
                unique_rows.append(row)
                seen_operators.add(operators)  # Mark this operators as seen

    # Save deduplicated content to new file
    with open(output_csv_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(unique_rows)


def split_operator_chains(input_csv_file, script_ops_csv_file, function_ops_csv_file):
    """
    Split operator sequences into script operators and custom function operators, and save to different CSV files.
    :param input_csv_file: Input CSV file path
    :param script_ops_csv_file: Script operators output CSV file path
    :param function_ops_csv_file: Function operators output CSV file path
    """
    df = pd.read_csv(input_csv_file)
    script_operators = []
    function_operators = []

    for _, row in df.iterrows():
        # Directly split string by comma
        operators = row['operators'].strip('"').split(',')

        current_script_ops = []
        current_function_ops = []
        function_depth = 0  # Used to track function nesting level
        current_function = None  # Current function being processed

        for op in operators:
            op = op.strip()

            # Handle function start marker
            if op == '[FUNCTION_START]':
                function_depth += 1
                if function_depth == 1:  # Only process at outermost function start
                    # Save previously collected script operators
                    if current_script_ops:
                        script_operators.append(','.join(current_script_ops))
                        current_script_ops = []
                continue

            # Handle function end marker
            if op == '[FUNCTION_END]':
                function_depth -= 1
                if function_depth == 0:  # Outermost function ends
                    # Save collected function operators
                    if current_function_ops:
                        function_operators.append(','.join(current_function_ops))
                        current_function_ops = []
                    current_function = None
                continue

            # Handle regular operators
            if function_depth > 0:  # Inside a function
                if op.startswith('[function]'):
                    # Remove [function] prefix
                    current_function_ops.append(op.replace('[function]', ''))
                elif not (op == '[FUNCTION_START]' or op == '[FUNCTION_END]'):
                    # Only add non-marker operators
                    current_function_ops.append(op)
            else:  # Outside functions
                if not (op == '[FUNCTION_START]' or op == '[FUNCTION_END]'):
                    current_script_ops.append(op)

        # Process remaining operators
        if current_script_ops:
            script_operators.append(','.join(current_script_ops))
        if current_function_ops:
            function_operators.append(','.join(current_function_ops))

    # Only clean empty strings, keep duplicates
    script_operators = [op for op in script_operators if op.strip()]
    function_operators = [op for op in function_operators if op.strip()]

    # Save script operators and function operators to different CSV files
    pd.DataFrame(script_operators, columns=['operators']).to_csv(script_ops_csv_file, index=False)
    pd.DataFrame(function_operators, columns=['operators']).to_csv(function_ops_csv_file, index=False)


# Script operators section
def get_all_operators_to_csv(input_csv, output_csv=None, key='operators'):
    df = pd.read_csv(input_csv)
    operators_set = set()

    for idx, row in df.iterrows():
        operators_list = ast.literal_eval(row[key])
        operators_set.update(operators_list)

    # Convert the set back to a sorted list
    unique_operators = sorted(list(operators_set))

    # Create DataFrame
    operators_df = pd.DataFrame(unique_operators, columns=["Unique_Operators"])
    if output_csv:
        operators_df.to_csv(output_csv, index=False)

    return unique_operators


def process_all_operators(df: pd.DataFrame, all_apis_set: Set[str], all_apis_short_set: Set[str],
        input_col: str = 'Operators', output_col: str = 'Processed_Operators') -> pd.DataFrame:
    """
    Standardize the operator list in a specific column of the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing the original operators.
        all_apis_set (set): Set of all standard APIs with full paths, e.g. {'ee.Image.Clip', ...}
        all_apis_short_set (set): Set of all standard APIs in short form, e.g. {'Clip', ...}
        input_col (str): Input column name, default is 'Operators'
        output_col (str): Output column name, default is 'Processed_Operators'

    Returns:
        pd.DataFrame: DataFrame with a new Processed_Operators column
    """

    def process_operator(operator: str) -> str:
        """Standardize a single operator"""
        if operator in all_apis_set:
            return operator

        short_name = operator.split('.')[-1]
        if short_name in all_apis_short_set:
            return short_name

        return "custom_operator"

    processed_operators_list = []

    for _, row in df.iterrows():
        raw_operators = row[input_col]

        try:
            # Convert string to list
            operators_list = [op.strip() for op in raw_operators.split(',')]
        except Exception as e:
            print(f"Error parsing row: {raw_operators}. Error: {e}")
            operators_list = []

        # Process each operator
        processed_list = [process_operator(op) for op in operators_list]
        processed_operators_list.append(processed_list)

    df[output_col] = processed_operators_list
    df = df.drop(columns=[input_col])

    return df


def index_operators(input_file: str, operators_file: str, output_dir: str):
    """
    将原始算子序列转换为索引序列。

    参数:
        input_file (str): 输入 CSV 文件路径，包含 'operators' 列。
        operators_file (str): 标准 API 文件路径，包含 full_name 和 short_name 两列。
        output_dir (str): 输出文件保存目录。
    """

    # Step 1: Read the standard API file
    all_apis = pd.read_csv(operators_file, encoding='utf-8')
    all_apis_set = set(all_apis['full_name'])
    all_apis_short_set = set(all_apis['short_name'])

    # Step 2: Read the input CSV file
    raw_df = pd.read_csv(input_file, encoding='utf-8')

    # Process the operators in the input DataFrame
    processed_df = process_all_operators(
        raw_df,
        all_apis_set=all_apis_set,
        all_apis_short_set=all_apis_short_set,
        input_col='operators',
        output_col='Processed_Operators'
    )

    # 保存处理后的序列
    processed_seq_path = os.path.join(output_dir, 'processed_operator_sequence_all.csv')
    processed_df.to_csv(processed_seq_path, index=False)
    print("已保存处理后的算子序列：", processed_seq_path)

    # Step 3: Get unique operators and create mapping
    # Flatten the list of lists to a single list
    all_processed_ops = [
        op
        for ops_list in processed_df['Processed_Operators']
        for op in ops_list
    ]

    unique_operators = sorted(set(all_processed_ops))

    # Create DataFrame and assign indices
    unique_operators_df = pd.DataFrame({
        'Processed_Operators': unique_operators,
        'Index': range(len(unique_operators))
    })

    # 保存 unique_operators
    unique_ops_path = os.path.join(output_dir, 'indexed_unique_operators.csv')
    unique_operators_df.to_csv(unique_ops_path, index=False)
    print("Saved operator mapping", unique_ops_path)

    # 构建字典：operator -> index
    operator_to_index: Dict[str, int] = dict(zip(
        unique_operators_df['Processed_Operators'],
        unique_operators_df['Index']
    ))

    # Step 4: 将 Processed_Operators 转换为索引序列
    def replace_with_index(op_list: List[str]) -> List[int]:
        return [operator_to_index.get(op, -1) for op in op_list if op != "custom_operator"]

    processed_df['Indexed_Operators'] = processed_df['Processed_Operators'].apply(replace_with_index)

    # 保存最终结果
    indexed_output_path = os.path.join(output_dir, 'indexed_operators_sequence_all.csv')
    # 删除原始的 Processed_Operators 列
    processed_df = processed_df.drop(columns=['Processed_Operators'])
    processed_df.to_csv(indexed_output_path, index=False)

    print(f"已生成索引序列文件：{indexed_output_path}")

    return operator_to_index


def convert_to_spmf_format(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.dropna()
    sequences = df['Indexed_Operators'].apply(lambda x: [i for i in ast.literal_eval(x) if i != 0]).tolist()

    # convert to SPMF format
    with open(output_file, 'w') as f:
        for i, sequence in enumerate(sequences):
            if sequence:
                # SPMF format: use space to separate numbers, and end with -1 -2
                f.write(' '.join(map(str, sequence)) + ' -1 -2\n')
            if i % 10000 == 0:
                print(f"Processed {i} sequences")
    print("Finished converting to spmf format")


def prefix_span_spmf(input_file: object, min_support: object, max_pattern_length: object, output_file: object) -> None:
    # Note: Need Java 21 or later installed and set in PATH
    print("Start mining frequent sequences")
    spmf = Spmf("PrefixSpan", input_filename=input_file,
                output_filename=r"./data/prefixspan_output.txt", arguments=[min_support, max_pattern_length],
                spmf_bin_location_dir=r"./")

    spmf.run()
    print("Spam complete!")
    print(spmf.to_pandas_dataframe(pickle=True))
    spmf.to_csv(output_file)


def span_to_operator(input_file, index_file, output_file):
    data_df = pd.read_csv(input_file, sep=';')
    index_df = pd.read_csv(index_file)
    # a dictionary to map index to operator
    index_dict = dict(zip(index_df['Index'], index_df['Processed_Operators']))
    def process_row(row):
        # remove brackets and quotes from the row
        row = row.strip("[]'\"")
        # try to convert the row into a list of integers
        try:
            span_list = list(map(int, row.split()))
        except ValueError:
            return None
        # remove list with only one element or all elements are the same
        if len(span_list) <= 1 or all(x == span_list[0] for x in span_list):
            return None
        try:
            word_list = [index_dict[num] for num in span_list]
        except KeyError:
            return None
        # return the processed list as a string
        return str(word_list)

    # Process the first column of the DataFrame
    data_df['processed'] = data_df.iloc[:, 0].apply(process_row)
    # Delete rows where 'processed' is None
    data_df = data_df.dropna(subset=['processed'])
    data_df = data_df[["processed"]]

    # Average sequence length calculation (optional)
    # total = 0
    # for index, row in data_df.iterrows():
    #     span_list = ast.literal_eval(row)
    #     total += len(span_list)
    # avg_seq_length = total/len(data_df) if len(data_df) > 0 else 0

    data_df.to_csv(output_file, index=False)

    print("Total sequences:", len(data_df))
    # print("Average sequence length:", avg_seq_length)


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
            print(f"Error parsing line{idx}: {e}")
            print(f"{row[key]}")
            sequences.append([])  # 添加空列表作为占位符

    return sequences, df


def save_lists_to_csv(lists, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for lst in lists:
            writer.writerow([str(lst)])


def analyze_pf_output_stats(file_path):
    df = pd.read_csv(file_path)
    df['processed'] = df['processed'].apply(ast.literal_eval)
    row_count = len(df)

    # 计算每个序列的长度
    lengths = df['processed'].apply(len)
    average_length = lengths.mean()

    # 统计每个长度对应的序列数目
    length_counts = lengths.value_counts().sort_index()

    print("File name:", file_path)
    print(f"Total sequences: {row_count}")
    print(f"Average sequence length: {average_length:.2f}")
    print("\nSequence length distribution:")

    # 输出每个长度的序列数目
    print("Length, Sequence Count")
    for length, count in length_counts.items():
        print(f"{length},{count}")


# Custom function section
def extract_custom_functions_from_file(file_path):
    """
    Extract custom functions from a single file.
    :param file_path: File path
    :return: List of extracted custom functions
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    # Regular expression to match custom functions
    # Match lines starting with "function", followed by function name and parameters, until the corresponding closing brace
    function_pattern = re.compile(
        r'function\s+\w+\s*\(.*?\)\s*\{.*?\}',
        re.DOTALL
    )

    # Find all matching functions
    functions = function_pattern.findall(content)
    return functions


def save_functions_to_file(functions, output_file):
    """
    Save the function list to a file, each function wrapped with <function>...</function>.
    :param functions: List of functions
    :param output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for func in functions:
            file.write(f"<function>\n{func}\n</function>\n\n")


def process_custom_functions_in_directory(directory, output_dir, batch_size=10000):
    """
    Parse all .txt files in the directory, extract custom functions, and save them in batches.
    :param directory: Input directory path
    :param output_dir: Output directory path
    :param batch_size: Save results every batch_size files processed
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    all_functions = []  # Store all extracted functions
    file_count = 0  # Number of files processed
    batch_number = 1  # Current batch number

    # Traverse all files in the directory
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                try:
                    # Extract functions from the current file
                    functions = extract_custom_functions_from_file(file_path)
                    all_functions.extend(functions)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

                file_count += 1

                # Save results every batch_size files processed
                if file_count % batch_size == 0:
                    output_file = os.path.join(output_dir, f"functions_batch_{batch_number}.txt")
                    save_functions_to_file(all_functions, output_file)
                    print(f"Saved {len(all_functions)} functions to {output_file}")
                    all_functions.clear()  # Clear function list
                    batch_number += 1

    # Process remaining functions (less than one batch)
    if all_functions:
        output_file = os.path.join(output_dir, f"functions_batch_{batch_number}.txt")
        save_functions_to_file(all_functions, output_file)
        print(f"Saved {len(all_functions)} functions to {output_file}")


def process_code_list(code_list, output_file):
    all_operators = []

    for i, code in enumerate(code_list):
        try:
            # Parse code to generate AST
            ast_data = esprima.parseScript(code, {'tolerant': True, 'jsx': True})
            # print(ast_data)
            # Convert AST object to dict
            ast_dict = json.loads(json.dumps(ast_data, default=lambda obj: obj.__dict__))
            # Extract operators
            operators = extract_operators_from_ast_py(ast_dict, skip_custom_function=False)
            # Use index as identifier
            all_operators.append([f"code_{i}", operators])

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} code snippets")
        except Exception as e:
            # print(f"Error processing code snippet {i}: {e}")
            continue

    def save_operators_to_csv(operators_list, output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header, columns are Code_ID and Operators
            writer.writerow(["Code_ID", "Operators"])
            # Write each code snippet's ID and operator sequence
            for code_id, operators in operators_list:
                if operators:
                    writer.writerow([code_id, str(operators)])

    save_operators_to_csv(all_operators, output_file)
    print(f"Processed {len(code_list)} code snippets in total")
    return all_operators


def get_function_list(input_path):
    """
    Extract all custom functions from the specified directory. The directory should contain functions wrapped with <function>...</function>.
    """
    functions = []
    function_pattern = re.compile(r'<function>(.*?)</function>', re.DOTALL)

    all_files = os.listdir(input_path)
    for file in all_files:
        try:
            file_path = os.path.join(input_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            functions.extend(function_pattern.findall(content))
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")

    return functions


def operator_sequence_clean_and_index(input_file, gee_operators_file, cleaned_output_file, encoded_output_file,
                                      mapping_output_file):
    def convert_string_to_sequence(input_file):
        df = pd.read_csv(input_file)
        sequences = []

        # Assume the sequence is in the column named 'Operators'
        for idx, row in df.iterrows():
            try:
                raw_operators = row['operators']
                sequence = [op.strip() for op in raw_operators.split(',')]
                sequences.append(sequence)
            except (SyntaxError, ValueError) as e:
                print(f"Error processing row {idx}: {e}")
                print(f"Problem string: {row['Operators']}")
                sequences.append([])  # Add empty list as placeholder

        return sequences, df

    def clean_sequences(sequences, gee_operators_file):
        # Read the official GEE operator list
        all_apis = pd.read_csv(gee_operators_file, encoding='utf-8')
        all_apis_full = all_apis['full_name']
        all_apis_short = all_apis['short_name']
        all_apis_set = set(all_apis_full)
        all_apis_short_set = set(all_apis_short)

        cleaned_sequences = []

        for sequence in sequences:
            cleaned_seq = []
            for op in sequence:
                if op in all_apis_set:
                    # If the full name is a GEE operator, keep it
                    cleaned_seq.append(op)
                else:
                    # Try splitting and take the last part
                    processed_operator = op.split('.')[-1]

                    if processed_operator in all_apis_short_set:
                        # If the last part is a GEE operator, keep that part
                        cleaned_seq.append(processed_operator)
                    else:
                        # Otherwise, rename as custom_operator
                        cleaned_seq.append("custom_operator")

            cleaned_sequences.append(cleaned_seq)

        return cleaned_sequences

    def encode_sequences(cleaned_sequences):
        # Find all unique operators
        all_operators = set()
        for seq in cleaned_sequences:
            all_operators.update(seq)

        # Create encoding mapping, custom_operator is 0
        operator_mapping = {"custom_operator": 0}
        counter = 1

        for op in sorted(all_operators):
            if op != "custom_operator":
                operator_mapping[op] = counter
                counter += 1

        # Encode sequences
        encoded_sequences = []
        for seq in cleaned_sequences:
            encoded_seq = [operator_mapping[op] for op in seq if op != "custom_operator"]
            encoded_sequences.append(encoded_seq)

        return encoded_sequences, operator_mapping

    # Step 1: Convert string to sequence
    sequences, original_df = convert_string_to_sequence(input_file)

    # Step 2: Clean sequences
    cleaned_sequences = clean_sequences(sequences, gee_operators_file)

    # Save cleaned sequences - using pandas
    result_df = original_df.copy()
    result_df['Processed_Operators'] = cleaned_sequences
    if 'Operators' in result_df.columns:
        result_df = result_df.drop(columns=['Operators'])
    result_df = result_df.dropna()
    result_df.to_csv(cleaned_output_file, index=False)
    print(f"Saved processed operator sequences: {cleaned_output_file}")

    # Step 3: Encode sequences
    encoded_sequences, operator_mapping = encode_sequences(cleaned_sequences)

    # Save encoded sequences - using pandas
    encoded_df = original_df.copy()
    encoded_df['Indexed_Operators'] = encoded_sequences
    encoded_df = encoded_df[['Indexed_Operators']]
    encoded_df.to_csv(encoded_output_file, index=False)
    print(f"Generated index sequence file: {encoded_output_file}")

    # Save operator mapping - using pandas
    mapping_df = pd.DataFrame({
        'Processed_Operators': [op for op, _ in sorted(operator_mapping.items(), key=lambda x: x[1])],
        'Index': [code for _, code in sorted(operator_mapping.items(), key=lambda x: x[1])]
    })
    mapping_df.to_csv(mapping_output_file, index=False)
    print(f"Saved unique operator mapping: {mapping_output_file}")


def get_custom_function_frequency(input_file, mapping_file, key="Indexed_Operators"):
    """
    Count the frequency of exactly identical custom functions from the input file, and map the number sequence back to operator names.
    :param input_file: CSV file path containing Indexed_Operators column
    :param mapping_file: Mapping file path, containing Index and Processed_Operators columns
    :param key: Column name containing the sequence, default is "Indexed_Operators"
    :return: List of operator sequences and their frequencies, sorted by frequency descending
    """
    # 1. Read mapping file, build Index -> Operator mapping
    df_map = pd.read_csv(mapping_file)
    index_to_op = dict(zip(df_map['Index'], df_map['Processed_Operators']))

    # 2. Read main data file
    df = pd.read_csv(input_file)
    print(len(df))
    sequences = []

    for idx, row in df.iterrows():
        try:
            # Use ast.literal_eval to convert string to Python list
            sequence = ast.literal_eval(row[key])
            if len(sequence) >= 2:
                # Convert to operator name sequence
                op_sequence = tuple([index_to_op[i] for i in sequence if i in index_to_op])
                sequences.append(op_sequence)
        except (ValueError, SyntaxError, KeyError):
            continue

    # 3. Count frequency
    function_frequency = Counter(sequences)

    # 4. Sort by frequency
    sorted_frequency = sorted(function_frequency.items(), key=lambda x: x[1], reverse=True)

    return sorted_frequency


def save_frequency_to_csv(freq_list, output_file, min_freq=1):
    """
    Save the operator sequence frequency list to a CSV file
    :param freq_list: Frequency list, format [(sequence_tuple, count), ...]
    :param output_file: Output CSV file path
    :param min_freq: Minimum frequency, sequences below this value will not be saved
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Sequence', 'Frequency'])

        # Write data
        for seq, freq in freq_list:
            if freq >= min_freq:
                writer.writerow([str(seq), freq])


def find_elbow(x, y):
    # Calculate the distance from each point to the line connecting the first and last points
    first = np.array([x[0], y[0]])
    last = np.array([x[-1], y[-1]])
    line_vec = last - first
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))

    vec_from_first = np.array([(x - first[0]), (y - first[1])])

    # Calculate the distance from each point to the line
    distances = np.abs(np.cross(line_vec_norm, vec_from_first.T))

    # Find the point with the maximum distance
    elbow_idx = np.argmax(distances)
    return x[elbow_idx], y[elbow_idx]


def plot_frequency_distribution(freq_list, start_threshold=2):
    """
    Plot a line chart showing the number of function sequences with frequency greater than or equal to x as x varies, using logarithmic coordinates.
    :param freq_list: Frequency list in the format [(sequence, count), ...], already sorted
    :param start_threshold: Starting threshold, default starts from 2
    """
    # Extract all frequencies
    counts = [count for _, count in freq_list]

    # Prepare data: x is the threshold, y is the number of sequences with count >= x
    max_freq = max(counts)
    x_values = list(range(start_threshold, max_freq + 1))
    y_values = []

    for threshold in x_values:
        # num_sequences = sum(1 for count in counts if count >= threshold)
        num_sequences = sum(1 for count in counts if count == threshold)
        y_values.append(num_sequences)

    elbow_x, elbow_y = find_elbow(x_values, y_values)
    print(elbow_x, elbow_y)

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
    plt.title('Number of Function Sequences with Frequency')
    plt.xlabel('Frequency Threshold (X)')
    plt.ylabel('Number of Sequences (log scale)')
    # plt.yscale('log')  # Use logarithmic Y axis
    plt.grid(True, which="both", ls="--")  # Show both major and minor grid lines
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Frequent sequence mining for script operators

    # Parse script operators from code
    folder_path = '../ScriptAST/script_ast'
    output_dir = './data/script_operators/'
    output_file = './data/script_operators/operators_sequence_script_and_function_all.csv'
    process_all_zips(folder_path, output_dir, output_file)
    merge_results(output_dir, output_file)

    split_operator_chains('./data/script_operators/operators_sequence_script_and_function_all.csv',
                          './data/script_operators/operators_sequence_all.csv',
                          './data/custom_functions/custom_function_operators_sequence_all.csv')

    # Index operators
    input_csv = './data/script_operators/operators_sequence_all.csv'
    indexed_operators_output_dir = './data/script_operators'
    operators_file = './data/all_GEE_APIs_utf8.csv'
    index_operators(input_csv, operators_file, indexed_operators_output_dir)

    # Process custom functions
    input_file = "./data/custom_functions/custom_function_operators_sequence_all.csv"
    gee_operators_file = "./data/all_GEE_APIs_utf8.csv"
    cleaned_output_file = "./data/custom_functions/custom_function_cleaned_sequences_all.csv"
    encoded_output_file = "./data/custom_functions/custom_function_indexed_sequences_all.csv"
    mapping_output_file = "./data/custom_functions/custom_function_operator_mapping.csv"
    operator_sequence_clean_and_index(input_file, gee_operators_file, cleaned_output_file, encoded_output_file, mapping_output_file)

    # Deduplicate custom functions
    input_path = r"../ScriptAST/custom_functions"
    js_func_list = get_function_list(input_path)
    js_func_list = list(set(js_func_list))
    print(len(js_func_list))

    # Frequent sequence mining for script operator sequences
    # NOTE: Requires Java 21 or higher
    convert_to_spmf_format('./data/script_operators/indexed_operators_sequence_all.csv',
                            './data/script_operators/sequences_spmf.txt')
    support_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    for support in support_list:
        prefix_span_spmf('./data/script_operators/sequences_spmf.txt', support, 10,
                         f"./data/script_operators/prefixspan_output_{support}_10.csv")
        span_to_operator(f'./data/script_operators/prefixspan_output_{support}_10.csv',
                         './data/script_operators/indexed_unique_operators.csv',
                         f'./data/script_operators/prefixspan_output_operators_{support}_10.csv')
        analyze_pf_output_stats(f'./data/script_operators/prefixspan_output_operators_{support}_10.csv')

    # Frequent sequence mining for custom function operator sequences
    # NOTE: Requires Java 21 or higher
    convert_to_spmf_format('./data/custom_functions/custom_function_indexed_sequences_all.csv',
                           './data/custom_functions/custom_functions_sequences_spmf.txt')
    support_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    for support in support_list:
        prefix_span_spmf('./data/custom_functions/custom_functions_sequences_spmf.txt', support, 10,
                         f"./data/custom_functions/custom_functions_prefixspan_output_{support}_10.csv")
        span_to_operator(f'./data/custom_functions/custom_functions_prefixspan_output_{support}_10.csv',
                         './data/custom_functions/custom_function_operator_mapping.csv',
                         f'./data/custom_functions/custom_function_prefixspan_output_operators_{support}_10.csv')
        analyze_pf_output_stats(f'./data/custom_functions/custom_function_prefixspan_output_operators_{support}_10.csv')

    # Mine frequently occurring custom functions
    seq_file = "./data/custom_functions/custom_function_indexed_sequences_all.csv"
    mapping_file = "./data/custom_functions/custom_function_operator_mapping.csv"
    freq_list = get_custom_function_frequency(seq_file, mapping_file)
    save_frequency_to_csv(freq_list, "./data/custom_functions/custom_fuction_frequency.csv", min_freq=2)
    plot_frequency_distribution(freq_list, start_threshold=2)

    # Merge files
    merge_csv_files(r'./data/script_operators/prefixspan_output_operators_0.1_10.csv',
                    r'./data/custom_functions/custom_function_prefixspan_output_operators_0.04_10.csv',
                    r'./data/prefix_span_combined_operators_list_0.1_0.04.csv')

    high_freq_func = pd.read_csv(r'./data/custom_functions/custom_fuction_frequency.csv')
    high_freq_func = high_freq_func[high_freq_func['Frequency'] >= 150]
    high_freq_func = high_freq_func[['Sequence']].rename(columns={'Sequence': 'processed'})
    high_freq_func.to_csv(r'./data/custom_functions/high_frequency_functions.csv', index=False)

    merge_csv_files(r"./data/custom_functions/high_frequency_functions.csv",
                    r'./data/prefix_span_combined_operators_list_0.1_0.04.csv',
                    r'./data/combined_operators_list_0.1_0.04_150.csv')
    analyze_pf_output_stats(f'./data/combined_operators_list_0.1_0.04_150.csv')

    # Count unique operators in merged file
    unique_operators = get_all_operators_to_csv("./data/prefix_span_combined_operators_list_0.25_0.1.csv", key="processed")
    print(len(unique_operators))
    print(unique_operators)

    print(1)
