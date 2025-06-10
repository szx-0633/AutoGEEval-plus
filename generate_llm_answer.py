import os
import shutil
import re
import yaml
import time
from typing import Optional, List, Dict, Tuple
import concurrent.futures
from tqdm import tqdm
from call_language_model import call_language_model


def extract_code_from_response_old(response_text: str) -> str:
    """
    从模型响应中提取Python代码

    参数:
    response_text (str): 模型的完整响应文本

    返回:
    str: 提取出的Python代码，如果没有找到则返回原始响应
    """
    # 尝试匹配三个反引号包围的代码块
    code_pattern = r"```python?(.*?)```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)

    if matches:
        # 返回第一个匹配的代码块，去除前后空白
        return matches[0].strip()
    else:
        # 如果没有找到代码块，返回原始响应
        return response_text.strip()


def filter_docstring(code_lines) -> list:
    result = []
    in_docstring = False  # 当前是否处于 docstring 中

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


def extract_code_from_response(response_text: str) -> str:
    """
    从模型响应中提取Python代码

    参数:
    response_text (str): 模型的完整响应文本

    返回:
    str: 提取出的Python代码，如果没有找到则返回原始响应
    """

    # 尝试匹配三个反引号包围的代码块
    code_pattern = r"```python?(.*?)```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)

    if matches:
        # 返回第一个匹配的代码块，去除前后空白
        extracted_lines = matches[0].strip().splitlines()
        filtered_lines = filter_docstring(extracted_lines)
        result = "\n".join(filtered_lines).strip()
        return result
    else:
        # 如果没有找到代码块，返回原始响应
        result = response_text.strip()
        return result


def process_test_file(
        test_file_path: str,
        output_dir: str,
        model_provider: str,
        model_name: str,
        stream: bool,
        system_prompt: str,
        temperature: Optional[float] = 0.2,
        max_tokens: Optional[int] = 2048,
        config_path: str = './llm_config.yaml',
        max_retries: int = 3,
        retry_delay: float = 2.0,
        type: str = "atomic"
) -> Dict:
    """
    Processes a single test file, calls the model, and saves the result.

    Args:
    test_file_path (str): Path to the test file.
    output_dir (str): Output directory.
    model_provider (str): Model provider.
    model_name (str): Model name.
    stream (bool): Whether to use streaming.
    system_prompt (str): System prompt.
    temperature (float, optional): Temperature parameter.
    max_tokens (int, optional): Maximum number of tokens to generate.
    config_path (str): Path to the configuration file.
    max_retries (int): Maximum number of retry attempts.
    retry_delay (float): Delay between retries in seconds.
    type (str): Type of requirement.

    Returns:
    Dict: Information about the processing result.
    """
    # Read the content of the test file
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_content = f.read()

    # Get the file name (without path and extension)
    file_name = os.path.basename(test_file_path)
    base_name = os.path.splitext(file_name)[0]

    output_file = os.path.join(output_dir, f"{base_name}_response.txt")

    # Create the prompt
    if type == "atomic":
        prompt = (
            "Please complete the following GEE Python API function, based on function signature and docstrings. "
            "Return ONLY the complete function code without any explanations or additional text. "
            "Do not add any comments beyond what's already in the docstring. Keep the given docstring. "
            "The input may be empty or invalid, you need to handle this situation appropriately to avoid program crashes. "
            "Warp the code with ```python and ``` to indicate the code block.\n\n"
            "Here is the function:\n\n"
            f"{test_content}"
        )
    else:
        prompt = (
            "Please complete the following GEE Python API function, based on function signature and docstrings. "
            "Return ONLY the complete function code without any explanations or additional text. "
            "Do NOT repeat the given docstring, but keep the function declaration. "
            "If you want to declare more functions, you MUST declare them INSIDE the given function. "
            "NEVER use `if __name__ == '__main__':` in your code. "
            "Warp the code with ```python and ``` to indicate the code block.\n\n"
            "Here is the function:\n\n"
            f"{test_content}"
        )
    # user_prompt = prompt_atomic
    user_prompt = prompt

    if "qwen3" in model_name:
        if "__thinking" in model_name:
            enable_thinking = True
            model_name = model_name.replace("__thinking", "")
            if model_provider != "ollama":
                stream = True
        else:
            enable_thinking = False
    else:
        enable_thinking = None

    if "qwq" in model_name:
        stream = True

    if model_name.startswith("o"):
        temperature = None
        max_tokens = None

    # 初始化重试相关变量
    retry_count = 0
    last_error = None
    response_text = None
    tokens_used = None
    error_msg = None

    while retry_count <= max_retries:
        try:
            # Call the model
            start_time = time.time()
            response_text, tokens_used, error_msg = call_language_model(
                model_provider=model_provider,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stream=stream,
                enable_thinking=enable_thinking,
                collect=True,
                temperature=temperature,
                max_tokens=max_tokens,
                files=None,
                config_path=config_path
            )
            elapsed_time = time.time() - start_time

            # 检查返回内容是否为空或存在错误
            if not response_text or error_msg:
                raise ValueError(f"Empty response or error: {error_msg}")

            # 如果成功获取到内容，跳出重试循环
            break

        except Exception as e:
            last_error = str(e)
            retry_count += 1

            # 如果达到最大重试次数，记录最后的错误
            if retry_count > max_retries:
                error_msg = f"Max retries ({max_retries}) exceeded. Last error: {last_error}"
                break

            # 等待一段时间后重试
            time.sleep(retry_delay * (1.5 ** (retry_count - 1)))  # 指数退避
            continue

    # Extract code
    code = response_text if response_text else ""

    # Save the result
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)

    # Return processing information
    result = {
        'test_file': test_file_path,
        'tokens_used': tokens_used,
        'elapsed_time': elapsed_time if 'elapsed_time' in locals() else None,
        'error': error_msg,
        'retry_count': retry_count,
        'success': response_text is not None and not error_msg
    }

    time.sleep(1)

    return result


def run_function_completion_tests(
        test_dir: str,
        models_config: List[Dict],
        type: str,
        stream: bool = False,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        config_path: str = './llm_config.yaml',
        parallel: bool = True,
        max_workers: int = 4,
        times: int = 5,
) -> Dict:
    """
    Runs function completion tests for all test files in the specified directory.

    Args:
    test_dir (str): Directory of test files.
    models_config (List[Dict]): List of model configurations, each containing provider and name.
    type (str): Type of requirement.
    system_prompt (str): System prompt.
    stream (bool): Whether to use streaming.
    temperature (float): Temperature parameter.
    max_tokens (int): Maximum number of tokens to generate.
    config_path (str): Path to the configuration file.
    parallel (bool): Whether to process in parallel.
    max_workers (int): Maximum number of parallel worker threads.
    times (int): Number of times to repeat the test.

    Returns:
    Dict: Test result statistics.
    """
    # Get all test files
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                  if os.path.isfile(os.path.join(test_dir, f)) and f.endswith('.txt')]
    test_file_count = len(test_files)

    results = {}

    for time_run in range(1, times + 1):
        for model_config in models_config:
            model_provider = model_config['provider']
            model_name = model_config['name']
            model_name_simple = model_config['name_simple']

            print(f"\nProcessing model: {model_provider}/{model_name_simple}")

            # Create output directory
            output_dir = f"./generate_results0/{model_name_simple}_{time_run}/{type}"
            os.makedirs(output_dir, exist_ok=True)
            output_file_count = len(os.listdir(output_dir))

            if output_file_count >= test_file_count:
                print(f"Model {model_provider}/{model_name_simple} has processed all files, skipping...")
                continue

            model_results = []
            if "qwen3" in model_name_simple and "thinking" in model_name_simple:
                model_name = model_name + "__thinking"

            if parallel:
                # Parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(
                            process_test_file,
                            test_file,
                            output_dir,
                            model_provider,
                            model_name,
                            stream,
                            system_prompt,
                            temperature,
                            max_tokens,
                            config_path,
                            type=type
                        ): test_file for test_file in test_files
                    }

                    for future in tqdm(concurrent.futures.as_completed(future_to_file),
                                       total=len(test_files),
                                       desc=f"Processing {model_name}"):
                        try:
                            result = future.result()
                            model_results.append(result)
                        except Exception as e:
                            test_file = future_to_file[future]
                            print(f"Error processing file {test_file}: {e}")
                            model_results.append({
                                'test_file': test_file,
                                'error': str(e)
                            })
            else:
                # Serial processing
                for test_file in tqdm(test_files, desc=f"Processing {model_name}"):
                    try:
                        result = process_test_file(
                            test_file,
                            output_dir,
                            model_provider,
                            model_name,
                            stream,
                            system_prompt,
                            temperature,
                            max_tokens,
                            config_path,
                            type=type
                        )
                        model_results.append(result)
                    except Exception as e:
                        print(f"Error processing file {test_file}: {e}")
                        model_results.append({
                            'test_file': test_file,
                            'error': str(e)
                        })

            # Calculate statistics
            total_files = len(model_results)
            successful = len([r for r in model_results if r.get('error') is None])
            total_tokens = sum(r.get('tokens_used', 0) for r in model_results if r.get('tokens_used') is not None)
            avg_time = sum(r.get('elapsed_time', 0) for r in model_results if r.get('elapsed_time') is not None) / max(
                successful, 1)

            # Save result summary
            summary = {
                'model_provider': model_provider,
                'model_name': model_name,
                'total_files': total_files,
                'successful': successful,
                'failed': total_files - successful,
                'total_tokens_used': total_tokens,
                'average_time_per_file': avg_time,
                'detailed_results': model_results
            }

            # Save summary to file
            summary_file = os.path.join(output_dir, "summary.yaml")
            with open(summary_file, 'w', encoding='utf-8') as f:
                yaml.dump(summary, f, default_flow_style=False)

            results[f"{model_provider}/{model_name}"] = summary

            print(f"Finished processing {model_provider}/{model_name_simple} for the {time_run}st/nd/rd/th time:")
            print(f"  Total files: {total_files}")
            print(f"  Successful: {successful}, Failed: {total_files - successful}")
            print(f"  Total tokens used: {total_tokens}")
            print(f"  Average processing time: {avg_time:.2f} seconds")

    return results


def clean_file(model_name: str):
    """Cleans the generated files for a given model."""
    raw_output_dir1 = f"./generate_results0/{model_name}/atomic/"
    raw_output_dir2 = f"./generate_results0/{model_name}/combined/"
    raw_output_dir3 = f"./generate_results0/{model_name}/theme/"
    output_dir1 = f"./generate_results/{model_name}/atomic/"
    output_dir2 = f"./generate_results/{model_name}/combined/"
    output_dir3 = f"./generate_results/{model_name}/theme/"

    raw_dirs = [raw_output_dir1, raw_output_dir2, raw_output_dir3]
    output_dirs = [output_dir1, output_dir2, output_dir3]

    for raw_output_dir, output_dir in zip(raw_dirs, output_dirs):
        # Check if the directory exists
        if not os.path.exists(raw_output_dir):
            print(f"Directory {raw_output_dir} does not exist, skipping cleaning for model {model_name}.")
            return

        os.makedirs(output_dir, exist_ok=True)

        generate_config_path = os.path.join(raw_output_dir, "summary.yaml")
        output_yaml_path = os.path.join(output_dir, "summary.yaml")
        all_new_stats = []

        with open (generate_config_path, 'r', encoding='utf-8') as f:
            generate_config = yaml.safe_load(f)

        for index, items in enumerate(generate_config['detailed_results']):
            test_file = items['test_file']
            test_file_name = os.path.basename(test_file)
            code_file_name = test_file_name.replace(".txt", "_response.txt")
            code_file_path = os.path.join(raw_output_dir, code_file_name)

            test_stats = {
                'model_name': model_name,
                'test_file': test_file,
                'tokens_used': items['tokens_used'],
                'elapsed_time': items['elapsed_time'],
                'error': items['error'],
                'raw_lines': 0,
                'cleaned_lines': 0,
            }

            with open(code_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            text = re.sub(r'<think>.*?</think>', '', file_content, flags=re.DOTALL)
            raw_num_lines = len(file_content.splitlines())

            # Process file content
            cleaned_content = clean_content(file_content)
            cleaned_num_lines = len(cleaned_content.splitlines())

            test_stats['raw_lines'] = raw_num_lines
            test_stats['cleaned_lines'] = cleaned_num_lines
            all_new_stats.append(test_stats)

            # Write back to file
            out_file_path = os.path.join(output_dir, code_file_name)
            with open(out_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

        with open(output_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(all_new_stats, f, default_flow_style=False)


def clean_content(content: str) -> str:
    """Cleans the provided content by removing unnecessary parts like think blocks and main function calls."""
    # 1. 删除<think>...</think>之间的内容
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    # 2. 提取所有代码块
    code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)

    if not code_blocks:
        # 如果没有找到代码块，返回清理后的全部内容
        return text.strip()

    # 3. 找到最后一个包含函数定义的代码块
    last_def_block = None
    for block in code_blocks:
        if 'def ' in block:
            last_def_block = block

    if not last_def_block:
        # 如果没有找到包含函数定义的代码块，返回最后一个代码块
        last_def_block = code_blocks[-1].strip()

    last_def_block = "\n".join(filter_docstring(last_def_block.splitlines()))

    # 4. 处理主函数定义
    if '__name__ == "__main__"' in last_def_block or "__name__ == '__main__'" in last_def_block:
        # 分割代码块并保留主函数之前的部分
        main_split = re.split(r'if\s+__name__\s*==\s*["\']__main__[\'"]\s*:', last_def_block)
        return main_split[0].strip()

    return last_def_block.strip()


if __name__ == "__main__":
    # System prompt
    system_prompt = (
        "You are an expert in geospatial analysis with python and Google Earth Engine(GEE)."
    )

    # Model configurations to test
    models_to_test = [
        {'provider': 'ollama', 'name': 'qwen2.5:3b', 'name_simple': 'qwen2.5_3b'},
        {'provider': 'ollama', 'name': 'qwen2.5:7b', 'name_simple': 'qwen2.5_7b'},
        {'provider': 'ollama', 'name': 'qwen2.5:32b', 'name_simple': 'qwen2.5_32b'},
        {'provider': 'ollama', 'name': 'qwen3:4b', 'name_simple': 'qwen3_4b'},
        {'provider': 'ollama', 'name': 'qwen3:8b', 'name_simple': 'qwen3_8b'},
        {'provider': 'ollama', 'name': 'qwen3:4b', 'name_simple': 'qwen3_4b_thinking'},
        {'provider': 'ollama', 'name': 'qwen3:8b', 'name_simple': 'qwen3_8b_thinking'},
        {'provider': 'ollama', 'name': 'qwen2.5-coder:3b', 'name_simple': 'qwen2.5_coder_3b'},
        {'provider': 'ollama', 'name': 'qwen2.5-coder:7b', 'name_simple': 'qwen2.5_coder_7b'},
        {'provider': 'ollama', 'name': 'qwen2.5-coder:32b', 'name_simple': 'qwen2.5_coder_32b'},
        {'provider': 'ollama', 'name': 'codellama:7b', 'name_simple': 'codellama_7b'},
        {'provider': 'ollama', 'name': 'deepseek-coder-v2:16b', 'name_simple': 'deepseek_coder_v2_16b'},
        {'provider': 'ollama', 'name': 'geocode-gpt:latest', 'name_simple': 'geocode_gpt'},
        {'provider': 'aliyun', 'name': 'qwq-32b', 'name_simple': 'qwq_32b'},
        {'provider': 'aliyun', 'name': 'qwen3-32b', 'name_simple': 'qwen3_32b'},
        {'provider': 'aliyun', 'name': 'qwen3-32b', 'name_simple': 'qwen3_32b_thinking'},
        {'provider': 'volcengine', 'name': 'deepseek-r1-250120', 'name_simple': 'deepseek_r1'},
        {'provider': 'volcengine', 'name': 'deepseek-v3-241226', 'name_simple': 'deepseek_v3_241226'},
        {'provider': 'volcengine', 'name': 'deepseek-v3-250324', 'name_simple': 'deepseek_v3_250324'},
        {'provider': 'openrouter', 'name': 'openai/gpt-4.1-mini', 'name_simple': 'gpt_4.1_mini'},
        {'provider': 'openrouter', 'name': 'openai/gpt-4.1', 'name_simple': 'gpt_4.1'},
        {'provider': 'openrouter', 'name': 'openai/o4-mini', 'name_simple': 'o4_mini'},
        {'provider': 'openrouter', 'name': 'anthropic/claude-3.7-sonnet', 'name_simple': 'claude_3.7_sonnet'},
        {'provider': 'openrouter', 'name': 'google/gemini-2.5-flash-preview-05-20', 'name_simple': 'gemini_2.5_flash_250520'},
    ]

    # Run tests
    results1 = run_function_completion_tests(
        test_dir="./test_code/atomic_code/test_instructions",
        models_config=models_to_test,
        type="atomic",
        stream=False,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=4096,
        config_path='./llm_config.yaml',
        max_workers=4,
        parallel=True,
        times=5
    )

    # Output summary
    print("\nGeneration complete!")
    for model, summary in results1.items():
        print(f"\n{model}:")
        print(
            f"  Success rate: {summary['successful']}/{summary['total_files']} ({summary['successful'] / summary['total_files'] * 100:.1f}%)")
        print(f"  Total tokens used: {summary['total_tokens_used']}")
        print(f"  Average processing time: {summary['average_time_per_file']:.2f} seconds")

    results2 = run_function_completion_tests(
        test_dir="./test_code/combined_code/test_instructions",
        models_config=models_to_test,
        type="combined",
        stream=False,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=4096,
        config_path='./llm_config.yaml',
        max_workers=4,
        parallel=True,
        times=5
    )
    
    # Output summary
    print("\nGeneration complete!")
    for model, summary in results2.items():
        print(f"\n{model}:")
        print(
            f"  Success rate: {summary['successful']}/{summary['total_files']} ({summary['successful'] / summary['total_files'] * 100:.1f}%)")
        print(f"  Total tokens used: {summary['total_tokens_used']}")
        print(f"  Average processing time: {summary['average_time_per_file']:.2f} seconds")


    results3 = run_function_completion_tests(
        test_dir="./test_code/theme_code/test_instructions",
        models_config=models_to_test,
        type="theme",
        stream=False,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=4096,
        config_path='./llm_config.yaml',
        max_workers=4,
        parallel=True,
        times=5
    )

    # Output summary
    print("\nGeneration complete!")
    for model, summary in results3.items():
        print(f"\n{model}:")
        print(
            f"  Success rate: {summary['successful']}/{summary['total_files']} ({summary['successful'] / summary['total_files'] * 100:.1f}%)")
        print(f"  Total tokens used: {summary['total_tokens_used']}")
        print(f"  Average processing time: {summary['average_time_per_file']:.2f} seconds")


    # Clean generated code files
    models_to_clean = [
        "gpt_4.1_1", "gpt_4.1_2", "gpt_4.1_3", "gpt_4.1_4", "gpt_4.1_5",
        "gpt_4.1_mini_1", "gpt_4.1_mini_2", "gpt_4.1_mini_3", "gpt_4.1_mini_4", "gpt_4.1_mini_5",
        "o4_mini_1", "o4_mini_2", "o4_mini_3", "o4_mini_4", "o4_mini_5",
        "qwen2.5_3b_1", "qwen2.5_3b_2", "qwen2.5_3b_3", "qwen2.5_3b_4", "qwen2.5_3b_5",
        "qwen2.5_7b_1", "qwen2.5_7b_2", "qwen2.5_7b_3", "qwen2.5_7b_4", "qwen2.5_7b_5",
        "qwen2.5_32b_1", "qwen2.5_32b_2", "qwen2.5_32b_3", "qwen2.5_32b_4", "qwen2.5_32b_5",
        "qwen3_4b_1", "qwen3_4b_2", "qwen3_4b_3", "qwen3_4b_4", "qwen3_4b_5",
        "qwen3_8b_1", "qwen3_8b_2", "qwen3_8b_3", "qwen3_8b_4", "qwen3_8b_5",
        "qwen3_32b_1", "qwen3_32b_2", "qwen3_32b_3", "qwen3_32b_4", "qwen3_32b_5",
        "qwen3_4b_thinking_1", "qwen3_4b_thinking_2", "qwen3_4b_thinking_3", "qwen3_4b_thinking_4", "qwen3_4b_thinking_5",
        "qwen3_8b_thinking_1", "qwen3_8b_thinking_2", "qwen3_8b_thinking_3", "qwen3_8b_thinking_4", "qwen3_8b_thinking_5",
        "qwen3_32b_thinking_1", "qwen3_32b_thinking_2", "qwen3_32b_thinking_3", "qwen3_32b_thinking_4", "qwen3_32b_thinking_5",
        "qwen2.5_coder_3b_1", "qwen2.5_coder_3b_2", "qwen2.5_coder_3b_3", "qwen2.5_coder_3b_4", "qwen2.5_coder_3b_5",
        "qwen2.5_coder_7b_1", "qwen2.5_coder_7b_2", "qwen2.5_coder_7b_3", "qwen2.5_coder_7b_4", "qwen2.5_coder_7b_5",
        "qwen2.5_coder_32b_1", "qwen2.5_coder_32b_2", "qwen2.5_coder_32b_3", "qwen2.5_coder_32b_4", "qwen2.5_coder_32b_5",
        "codellama_7b_1", "codellama_7b_2", "codellama_7b_3", "codellama_7b_4", "codellama_7b_5",
        "deepseek_coder_v2_16b_1", "deepseek_coder_v2_16b_2", "deepseek_coder_v2_16b_3", "deepseek_coder_v2_16b_4", "deepseek_coder_v2_16b_5",
        "geocode_gpt_1", "geocode_gpt_2", "geocode_gpt_3", "geocode_gpt_4", "geocode_gpt_5",
        "qwq_32b_1", "qwq_32b_2", "qwq_32b_3", "qwq_32b_4", "qwq_32b_5",
        "deepseek_r1_1", "deepseek_r1_2", "deepseek_r1_3", "deepseek_r1_4", "deepseek_r1_5",
        "deepseek_v3_241226_1", "deepseek_v3_241226_2", "deepseek_v3_241226_3", "deepseek_v3_241226_4", "deepseek_v3_241226_5",
        "deepseek_v3_250324_1", "deepseek_v3_250324_2", "deepseek_v3_250324_3", "deepseek_v3_250324_4", "deepseek_v3_250324_5",
        "claude_3.7_sonnet_1", "claude_3.7_sonnet_2", "claude_3.7_sonnet_3", "claude_3.7_sonnet_4", "claude_3.7_sonnet_5",
        "gemini_2.5_flash_250520_1", "gemini_2.5_flash_250520_2", "gemini_2.5_flash_250520_3", "gemini_2.5_flash_250520_4", "gemini_2.5_flash_250520_5",
    ]
    for model in models_to_clean:
        clean_file(model)
        print("Cleaned all files for model: ", model)

    print(1)
