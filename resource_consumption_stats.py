import os
import yaml
import re
import pandas as pd

def calculate_resource_consumption_stats(model_basenames, task_types):
    all_stats = {}

    for task in task_types:
        print(f"\n=== Task Type: {task} ===")
        print("Model,Avg_Tokens,Avg_Time(s),Avg_Cleaned_Lines,Avg_Raw_Lines")
        print("-" * 80)

        task_stats = {}

        for model in model_basenames:
            # Initialize statistics for the current model and current task
            task_stats[model] = {
                'avg_tokens': 0,
                'avg_time': 0,
                'avg_cleaned_lines': 0,
                'avg_raw_lines': 0
            }

            # Iterate over 5 runs (run_index from 1 to 5)
            found_valid_run = False
            for run_index in range(1, 6):
                filename = f"./generate_results/{model}_{run_index}/{task}/summary.yaml"

                if os.path.exists(filename):
                    try:
                        with open(filename, 'r', encoding='utf-8') as file:
                            data = yaml.safe_load(file)

                        if data:  # Ensure data is not empty
                            # Collect all results for the current run
                            run_tokens = []
                            run_times = []
                            run_cleaned_lines = []
                            run_raw_lines = []

                            for item in data:
                                tokens = item.get('tokens_used', 0)
                                time = item.get('elapsed_time', 0)
                                cleaned_lines = item.get('cleaned_lines', 0)
                                raw_lines = item.get('raw_lines', 0)

                                run_tokens.append(tokens)
                                run_times.append(time)
                                run_cleaned_lines.append(cleaned_lines)
                                run_raw_lines.append(raw_lines)

                            # Calculate the average for the current run
                            if run_tokens:
                                avg_tokens = sum(run_tokens) / len(run_tokens)
                                avg_time = sum(run_times) / len(run_times)
                                avg_cleaned_lines = sum(run_cleaned_lines) / len(run_cleaned_lines)
                                avg_raw_lines = sum(run_raw_lines) / len(run_raw_lines)

                                # If the average token count is greater than 0, use this result
                                if avg_tokens > 0:
                                    task_stats[model]['avg_tokens'] = avg_tokens
                                    task_stats[model]['avg_time'] = avg_time
                                    task_stats[model]['avg_cleaned_lines'] = avg_cleaned_lines
                                    task_stats[model]['avg_raw_lines'] = avg_raw_lines
                                    break  # Break after finding a valid run

                                task_stats[model]['avg_tokens'] = avg_tokens
                                task_stats[model]['avg_time'] = avg_time
                                task_stats[model]['avg_cleaned_lines'] = avg_cleaned_lines
                                task_stats[model]['avg_raw_lines'] = avg_raw_lines

                    except Exception as e:
                        print(f"Warning: Error reading {filename}: {e}")
                        continue

            # Output the statistics for the current model and current task
            model_stats = task_stats[model]
            print(f"{model},{model_stats['avg_tokens']:.2f},"
                  f"{model_stats['avg_time']:.4f},{model_stats['avg_cleaned_lines']:.2f},"
                  f"{model_stats['avg_raw_lines']:.2f}")

        # Save the statistics for the current task
        all_stats[task] = task_stats


def calculate_codeLines_from_raw(model_basenames, task_types):
    """
    Calculate code lines based on model name and task type.
    """
    for task in task_types:
        print(f"\n=== Task Type: {task} ===")
        print("Model,Avg_Cleaned_Lines,Avg_Raw_Lines")
        print("-" * 80)

        task_stats = {}
        for model in model_basenames:
            # Initialize statistics for the current model and current task
            task_stats[model] = {
                'avg_cleaned_lines': 0,
                'avg_raw_lines': 0
            }
            # Iterate over 5 runs (run_index from 1 to 5)
            total_cleaned_lines = 0
            total_raw_lines = 0
            total_files = 0
            for run_index in range(1, 6):
                files_dir = f"./generate_results0/{model}_{run_index}/{task}"
                files = os.listdir(files_dir)

                for file in files:
                    if file.endswith(".txt"):
                        with open(os.path.join(files_dir, file), 'r', encoding='utf-8') as f:
                            content = f.read()

                            cleaned_content = extract_code_from_response(content)
                            total_raw_lines += len(content.splitlines())
                            total_cleaned_lines += len(cleaned_content.splitlines())
                            total_files += 1

            # Calculate averages
            if total_files > 0:
                task_stats[model]['avg_cleaned_lines'] = total_cleaned_lines / total_files
                task_stats[model]['avg_raw_lines'] = total_raw_lines / total_files
            else:
                task_stats[model]['avg_cleaned_lines'] = 0
                task_stats[model]['avg_raw_lines'] = 0

            model_stats = task_stats[model]
            print(f"{model},{model_stats['avg_cleaned_lines']:.2f},"
                  f"{model_stats['avg_raw_lines']:.2f}")


def extract_code_from_response(response_text: str) -> str:
    """
    Extract Python code from model response

    Args:
    response_text (str): The full response text from the model

    Returns:
    str: Extracted Python code, or the original response if not found
    """

    def filter_docstring(code_lines) -> list:
        result = []
        in_docstring = False  # Whether currently inside a docstring

        for line in code_lines:
            quotes_in_line = line.count('"""')
            if not in_docstring:
                if quotes_in_line == 1:
                    in_docstring = True
                    continue
                elif quotes_in_line == 2:
                    continue
                else:
                    result.append(line)
            else:
                if quotes_in_line == 1:
                    in_docstring = False
                    continue
                else:
                    continue
        return result

    # Try to match code blocks surrounded by triple backticks
    code_pattern = r"```python?(.*?)```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)

    if matches:
        # Return the first matched code block, stripping leading/trailing whitespace
        extracted_lines = matches[0].strip().splitlines()
        filtered_lines = filter_docstring(extracted_lines)
        result = "\n".join(filtered_lines).strip()
        return result
    else:
        # If no code block is found, return the original response
        result = response_text.strip()
        return result


def calculate_resource_consumption_by_test_cases(model_basenames, task_types):
    """
    Calculate resource consumption statistics at the test case level. For multiple test cases per task, duplicate accordingly.
    """

    def python_constructor(loader, node):
        """
        Custom constructor for parsing Python code in !python tags, no processing needed in this file.
        """
        return None  # Return None if there is no function

    yaml.add_constructor('!python', python_constructor)

    for task in task_types:
        print(f"\n=== Task Type: {task} ===")
        print("Model,Avg_Tokens,Avg_Time(s),Avg_Cleaned_Lines,Avg_Raw_Lines")
        print("-" * 80)

        with open(f"./test_code/{task}_code/{task}_test_config.yaml", 'r', encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.Loader)

        for model in model_basenames:
            # Iterate over 5 runs (run_index from 1 to 6)
            for run_index in range(1, 7):
                # Initialize statistics for the current model and current task
                task_stats = {
                    'avg_tokens': 0,
                    'avg_time': 0,
                    'avg_cleaned_lines': 0,
                    'avg_raw_lines': 0
                }

                filename = f"./generate_results/{model}_{run_index}/{task}/summary.yaml"

                if not os.path.exists(filename):
                    continue

                if os.path.exists(filename):
                    try:
                        with open(filename, 'r', encoding='utf-8') as file:
                            data = yaml.safe_load(file)

                        if data:
                            # Collect all results for the current run
                            run_tokens = []
                            run_times = []
                            run_cleaned_lines = []
                            run_raw_lines = []

                            for item in data:
                                function_name = item.get('test_file', None).split('\\')[-1].replace('_test_instruction.txt', '')

                                test_cases_config = configs.get(function_name, [])

                                for _ in enumerate(test_cases_config):
                                    tokens = item.get('tokens_used', 0)
                                    time = item.get('elapsed_time', 0)
                                    cleaned_lines = item.get('cleaned_lines', 0)
                                    raw_lines = item.get('raw_lines', 0)

                                    run_tokens.append(tokens)
                                    run_times.append(time)
                                    run_cleaned_lines.append(cleaned_lines)
                                    run_raw_lines.append(raw_lines)

                            # Calculate the average for the current run
                            if run_tokens:
                                avg_tokens = sum(run_tokens) / len(run_tokens)
                                avg_time = sum(run_times) / len(run_times)
                                avg_cleaned_lines = sum(run_cleaned_lines) / len(run_cleaned_lines)
                                avg_raw_lines = sum(run_raw_lines) / len(run_raw_lines)

                                task_stats['avg_tokens'] = avg_tokens
                                task_stats['avg_time'] = avg_time
                                task_stats['avg_cleaned_lines'] = avg_cleaned_lines
                                task_stats['avg_raw_lines'] = avg_raw_lines

                    except Exception as e:
                        print(f"Warning: Error reading {filename}: {e}")
                        continue

                # Output the statistics for the current model and current task
                model_stats = task_stats
                # print(f"{model}_{run_index},{model_stats['avg_tokens']:.2f},"
                #       f"{model_stats['avg_time']:.4f},{model_stats['avg_cleaned_lines']:.2f},"
                #       f"{model_stats['avg_raw_lines']:.2f}")
                print(f"{model}_{run_index},{model_stats['avg_tokens']:.2f},"
                      f"{model_stats['avg_time']:.4f}")


def calculate_resource_consumption_from_raw_by_test_cases(model_basenames, task_types):
    """
    Calculate resource consumption and code lines at the test case level based on model name and task type.
    """
    def python_constructor(loader, node):
        """
        Custom constructor for parsing Python code in !python tags, no processing needed in this file.
        """
        return None  # Return None if there is no function

    yaml.add_constructor('!python', python_constructor)

    for task in task_types:
        print(f"\n=== Task Type: {task} ===")
        print("Model,Avg_Tokens,Avg_Time(s),Avg_Cleaned_Lines,Avg_Raw_Lines")
        print("-" * 80)

        task_stats = {}

        with open(f"./test_code/{task}_code/{task}_test_config.yaml", 'r', encoding='utf-8') as f:
            configs = yaml.load(f, Loader=yaml.Loader)

        for model in model_basenames:
            # Iterate over 5 runs (run_index from 1 to 6)
            all_avg_tokens = []
            all_avg_times = []
            all_avg_cleaned_lines = []
            all_avg_raw_lines = []

            for run_index in range(1, 7):
                # Initialize statistics for the current model and current task
                run_stats = {
                    'avg_tokens': 0,
                    'avg_time': 0,
                    'avg_cleaned_lines': 0,
                    'avg_raw_lines': 0
                }

                filename = f"./generate_results/{model}_{run_index}/{task}/summary.yaml"

                if not os.path.exists(filename):
                    continue

                if os.path.exists(filename):
                    try:
                        with open(filename, 'r', encoding='utf-8') as file:
                            data = yaml.safe_load(file)

                        if data:
                            # Collect all results for the current run
                            run_tokens = []
                            run_times = []
                            run_cleaned_lines = []
                            run_raw_lines = []

                            for item in data:
                                function_name = item.get('test_file', None).split('\\')[-1].replace('_test_instruction.txt', '')
                                test_cases_config = configs.get(function_name, [])

                                code_filename = function_name + "_test_instruction_response.txt"
                                code_file_path = os.path.join(f"./generate_results0/{model}_{run_index}/{task}", code_filename)
                                with open(code_file_path, 'r', encoding='utf-8') as file:
                                    content = file.read()
                                raw_lines = len(content.splitlines())
                                cleaned_content = extract_code_from_response(content)
                                cleaned_lines = len(cleaned_content.splitlines())

                                for _ in enumerate(test_cases_config):
                                    tokens = item.get('tokens_used', 0)
                                    time = item.get('elapsed_time', 0)

                                    run_tokens.append(tokens)
                                    run_times.append(time)
                                    run_cleaned_lines.append(cleaned_lines)
                                    run_raw_lines.append(raw_lines)

                            # Calculate the average for the current run
                            if run_tokens:
                                avg_tokens = sum(run_tokens) / len(run_tokens)
                                avg_time = sum(run_times) / len(run_times)
                                avg_cleaned_lines = sum(run_cleaned_lines) / len(run_cleaned_lines)
                                avg_raw_lines = sum(run_raw_lines) / len(run_raw_lines)

                                run_stats['avg_tokens'] = avg_tokens
                                run_stats['avg_time'] = avg_time
                                run_stats['avg_cleaned_lines'] = avg_cleaned_lines
                                run_stats['avg_raw_lines'] = avg_raw_lines

                                csv_data = {
                                    'run_tokens': run_tokens,
                                    'run_times': run_times,
                                    'run_cleaned_lines': run_cleaned_lines,
                                    'run_raw_lines': run_raw_lines
                                }
                                df = pd.DataFrame(csv_data)
                                os.makedirs(f"./resource_consumption/{model}_{run_index}", exist_ok=True)
                                detailed_csv_file = f"./resource_consumption/{model}_{run_index}/{task}_consumption.csv"
                                df.to_csv(detailed_csv_file, index=False)

                            all_avg_tokens.append(run_stats['avg_tokens'])
                            all_avg_times.append(run_stats['avg_time'])
                            all_avg_cleaned_lines.append(run_stats['avg_cleaned_lines'])
                            all_avg_raw_lines.append(run_stats['avg_raw_lines'])

                    except Exception as e:
                        print(f"Warning: Error reading {filename}: {e}")
                        continue

            task_stats[model] = {
                'avg_tokens': sum(all_avg_tokens) / len(all_avg_tokens) if all_avg_tokens else 0,
                'avg_time': sum(all_avg_times) / len(all_avg_times) if all_avg_times else 0,
                'avg_cleaned_lines': sum(all_avg_cleaned_lines) / len(all_avg_cleaned_lines) if all_avg_cleaned_lines else 0,
                'avg_raw_lines': sum(all_avg_raw_lines) / len(all_avg_raw_lines) if all_avg_raw_lines else 0
            }

            model_stats = task_stats[model]
            print(f"{model},{model_stats['avg_tokens']:.2f},"
                f"{model_stats['avg_time']:.4f},{model_stats['avg_cleaned_lines']:.2f},"
                f"{model_stats['avg_raw_lines']:.2f}")


if __name__ == "__main__":
    model_basenames = ["gpt_4.1", "gpt_4.1_mini", "claude_3.7_sonnet", "deepseek_v3_241226", "deepseek_v3_250324",
                       "gemini_2.5_flash_250520", "qwen2.5_3b", "qwen2.5_7b", "qwen2.5_32b", "qwen3_4b", "qwen3_8b",
                       "qwen3_32b", "o4_mini", "qwq_32b", "qwen3_4b_thinking", "qwen3_8b_thinking", "qwen3_32b_thinking",
                       "deepseek_r1", "deepseek_coder_v2_16b", "qwen2.5_coder_3b", "qwen2.5_coder_7b", "qwen2.5_coder_32b",
                       "codellama_7b", "geocode_gpt"]
    task_types = ["atomic", "combined", "theme"]

    calculate_resource_consumption_from_raw_by_test_cases(model_basenames, task_types)

    print(1)

