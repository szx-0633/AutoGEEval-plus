import os
import json
import yaml
from pathlib import Path
from collections import defaultdict


def process_test_reports(reports_dir, output_yaml_path, configs):
    """
    Process the test report folder, merge into a YAML file and generate statistics

    Args:
        reports_dir: Path to the folder containing test reports
        output_yaml_path: Output merged YAML file path
        configs: Original test case configuration
    """

    def truncate_error_message(error_msg, max_chars=2000):
        """
        Truncate error message by character length limit

        Args:
            error_msg: Error message string
            max_chars: Maximum number of characters, default is 2000

        Returns:
            Truncated error message
        """
        if not isinstance(error_msg, str):
            error_msg = str(error_msg)

        if len(error_msg) <= max_chars:
            return error_msg

        # Truncate and add prompt message
        truncated_msg = error_msg[:max_chars]
        truncated_msg += f"\n\n... [Error message truncated, original length {len(error_msg)} characters, only showing first {max_chars} characters]"

        return truncated_msg

    # Dictionary to store all test reports, using function_name as the key
    all_reports = []

    # Traverse all JSON files in the folder
    total_count = 0
    for file_path in Path(reports_dir).glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                report = json.load(f)
                function_name = report["function_name"]
                file_errors = report.get("file_errors", [])

                if len(file_errors) > 0:
                    # If file_errors exist, load all test cases for this task from the config file, all marked as failed
                    test_cases_config = configs.get(function_name, [])

                    for i, test_case in enumerate(test_cases_config):
                        # Truncate error message
                        truncated_error = truncate_error_message(file_errors[0])

                        failed_test_case = {
                            "function_name": function_name,
                            "test_case_id": i + 1,
                            "edge_test": test_case["edge_test"],
                            "status": "failed",
                            "error": truncated_error,
                            "retry_count": 0,
                        }
                        all_reports.append(failed_test_case)

                else:
                    test_cases = report.get("test_cases", [])

                    for i, test_case in enumerate(test_cases):
                        # Set other necessary fields
                        test_case["function_name"] = function_name

                        # If there is also an error message in the test case, also truncate it
                        if "error" in test_case:
                            test_case["error"] = truncate_error_message(test_case["error"])

                        all_reports.append(test_case)

            except json.JSONDecodeError:
                print(f"Unable to parse JSON file: {file_path}")
                total_count += 1
                continue

            except Exception as e:
                print(f"Error occurred while processing file {file_path}: {e}")
                total_count += 1
                continue

            total_count += 1

    with open(output_yaml_path, "w", encoding="utf-8") as output_file:
        yaml.dump(all_reports, output_file, allow_unicode=True)

    print(f"Merged {total_count} test reports into {output_yaml_path}, total test cases: {len(all_reports)}")


def python_constructor(loader, node):
    """
    Custom constructor for parsing Python code in !python tags, no processing needed in this file.
    """
    return None  # Return None if there is no function


def process_models(models, task_types):
    yaml.add_constructor('!python', python_constructor)

    for model in models:
        for task_type in task_types:
            reports_directory = f"./generate_results/{model}/{task_type}_output/reports"  # Folder containing test reports
            merged_yaml_path = f"./generate_results/{model}/{task_type}_test_reports.yaml"  # Path to merged YAML file

            with open(f"./test_code/{task_type}_code/{task_type}_test_config.yaml", 'r', encoding='utf-8') as f:
                  config = yaml.load(f, Loader=yaml.Loader)

            process_test_reports(reports_directory, merged_yaml_path, config)


def classify_error_type(test_case):
    """
    Classify error type based on error message (only for non-edge test cases)
    """
    if test_case.get('status') != 'failed':
        return None

    # Do not classify errors for edge test cases
    if test_case.get('edge_test', False):
        return None

    error_msg = test_case.get('error', '').lower()

    # Determine error type based on error message (non-edge test cases only)
    if 'error executing code' in error_msg:
        error_type = 'syntax error'
    elif 'error in test case' in error_msg:
        error_type = 'attribute or parameter error'
    elif 'output type mismatch' in error_msg:
        error_type = 'output type mismatch'
    elif 'error getting download url' in error_msg or 'error when checking' in error_msg:
        error_type = 'wrong answer'
    elif 'timed out' in error_msg:
        error_type = 'runtime error'
    elif 'network error' in error_msg or 'max retries' in error_msg or 'error downloading' in error_msg:
        error_type = 'network error'
    else:
        print(f"Unclassified error message: {error_msg}")
        error_type = 'other error'

    return error_type


def calculate_single_run_pass_rate(model, task_type, run_index):
    """
    Calculate the pass rate for a single run
    """
    atomic_total_tests = 5078
    combined_total_tests = 1199
    theme_total_tests = 88

    total_tests_map = {
        'atomic': atomic_total_tests,
        'combined': combined_total_tests,
        'theme': theme_total_tests
    }

    total_tests = total_tests_map.get(task_type, 0)
    if total_tests == 0:
        return 0.0

    yaml_file = f"./generate_results/{model}_{run_index}/{task_type}_test_reports.yaml"
    yaml_path = Path(yaml_file)

    if not yaml_path.exists():
        print(f"Warning: File does not exist {yaml_file}")
        return 0.0

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            test_reports = yaml.safe_load(f) or []

        # Count the number of passed test cases
        passed_tests = sum(1 for test_case in test_reports
                           if test_case.get('status') == 'passed')

        return passed_tests / total_tests

    except Exception as e:
        print(f"Error reading file {yaml_file}: {e}")
        return 0.0


def calculate_pass_at_k(model, task_type, k):
    """
    Calculate pass@k rate
    """
    atomic_total_tests = 5078
    combined_total_tests = 1199
    theme_total_tests = 88

    total_tests_map = {
        'atomic': atomic_total_tests,
        'combined': combined_total_tests,
        'theme': theme_total_tests
    }

    total_tests = total_tests_map.get(task_type, 0)
    if total_tests == 0:
        return 0.0

    # Store the results of each test case (function_name, test_case_id) -> [status1, status2, ...]
    test_case_results = defaultdict(list)

    # Read the results of k runs
    for run_index in range(1, k + 1):
        yaml_file = f"./generate_results/{model}_{run_index}/{task_type}_test_reports.yaml"
        yaml_path = Path(yaml_file)

        if not yaml_path.exists():
            continue

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                test_reports = yaml.safe_load(f) or []

            # Process the results of each test case
            for test_case in test_reports:
                function_name = test_case.get('function_name', '')
                test_case_id = test_case.get('test_case_id', 0)
                status = test_case.get('status', 'failed')

                test_key = (function_name, test_case_id)
                test_case_results[test_key].append(status == 'passed')

        except Exception as e:
            continue

    # Calculate pass@k
    passed_tests = 0
    for test_key, results in test_case_results.items():
        if len(results) > 0 and any(results):
            passed_tests += 1

    return passed_tests / total_tests if total_tests > 0 else 0.0


def calculate_non_edge_error_distribution(model, task_type):
    """
    Calculate the error type distribution for non-edge test cases
    """
    error_counts = defaultdict(int)
    total_non_edge_failed = 0

    # Count error types for all 5 runs (non-edge test cases only)
    for run_index in range(1, 6):
        yaml_file = f"./generate_results/{model}_{run_index}/{task_type}_test_reports.yaml"
        yaml_path = Path(yaml_file)

        if not yaml_path.exists():
            continue

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                test_reports = yaml.safe_load(f) or []

            for test_case in test_reports:
                # Only count failed non-edge test cases
                if (test_case.get('status') == 'failed' and
                        not test_case.get('edge_test', False)):
                    total_non_edge_failed += 1
                    error_type = classify_error_type(test_case)
                    if error_type:
                        error_counts[error_type] += 1

        except Exception as e:
            continue

    # Calculate proportions
    error_proportions = {}
    if total_non_edge_failed > 0:
        for error_type, count in error_counts.items():
            error_proportions[error_type] = count / total_non_edge_failed

    return error_proportions


def calculate_edge_case_pass_rate(model, task_type):
    """
    Calculate the pass rate for edge test cases
    """
    edge_passed = 0
    edge_total = 0

    # Count the results of edge test cases for all 5 runs
    for run_index in range(1, 6):
        yaml_file = f"./generate_results/{model}_{run_index}/{task_type}_test_reports.yaml"
        yaml_path = Path(yaml_file)

        if not yaml_path.exists():
            continue

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                test_reports = yaml.safe_load(f) or []

            for test_case in test_reports:
                # Only count edge test cases
                if test_case.get('edge_test', False):
                    edge_total += 1
                    if test_case.get('status') == 'passed':
                        edge_passed += 1

        except Exception as e:
            continue

    # Calculate edge test case pass rate
    edge_pass_rate = edge_passed / edge_total if edge_total > 0 else 0.0

    return edge_pass_rate


def calculate_all_models_comprehensive_results(models, task_types):
    """
    Calculate comprehensive results for all models, including pass rate, non-edge error type distribution, and edge test case pass rate
    """
    # Define the order of non-edge error types
    non_edge_error_types = [
        'syntax error',
        'attribute or parameter error',
        'output type mismatch',
        'wrong answer',
        'runtime error',
        'network error',
        'other error'
    ]

    for task_type in task_types:
        print(f"\n=== {task_type.upper()} TASK TYPE RESULTS ===")

        # Build table header
        header = "Model,Run1,Run2,Run3,Run4,Run5,Average,Pass@1,Pass@3,Pass@5,Edge_Pass_Rate"
        for error_type in non_edge_error_types:
            header += f",{error_type}"
        print(header)

        for model in models:
            # Calculate pass rate for a single run
            single_runs = []
            sum_pass = 0.0
            for run_index in range(1, 6):
                single_rate = calculate_single_run_pass_rate(model, task_type, run_index)
                single_runs.append(f"{single_rate:.4f}")
                sum_pass += single_rate
            average = sum_pass / 5

            # Calculate pass@k
            pass_at_1 = calculate_pass_at_k(model, task_type, 1)
            pass_at_3 = calculate_pass_at_k(model, task_type, 3)
            pass_at_5 = calculate_pass_at_k(model, task_type, 5)

            # Calculate edge test case pass rate
            edge_pass_rate = calculate_edge_case_pass_rate(model, task_type)

            # Calculate non-edge error type distribution
            non_edge_error_distribution = calculate_non_edge_error_distribution(model, task_type)

            # Build output line
            result_line = f"{model},{','.join(single_runs)},{average:.4f},{pass_at_1:.4f},{pass_at_3:.4f},{pass_at_5:.4f},{edge_pass_rate:.4f}"

            # Add non-edge error type proportions
            for error_type in non_edge_error_types:
                proportion = non_edge_error_distribution.get(error_type, 0.0)
                result_line += f",{proportion:.4f}"

            print(result_line)


def calculate_detailed_edge_statistics(models, task_types):
    """
    Calculate detailed statistics for edge test cases
    """
    print(f"\n=== Detailed statistics for edge test cases ===")

    for task_type in task_types:
        print(f"\n--- {task_type.upper()} Edge Test Case Statistics ---")
        print("Model,Edge_Passed,Edge_Total,Edge_Pass_Rate")

        for model in models:
            edge_passed = 0
            edge_total = 0

            # Count the results of edge test cases for all 5 runs
            for run_index in range(1, 6):
                yaml_file = f"./generate_results/{model}_{run_index}/{task_type}_test_reports.yaml"
                yaml_path = Path(yaml_file)

                if not yaml_path.exists():
                    continue

                try:
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        test_reports = yaml.safe_load(f) or []

                    for test_case in test_reports:
                        if test_case.get('edge_test', False):
                            edge_total += 1
                            if test_case.get('status') == 'passed':
                                edge_passed += 1

                except Exception as e:
                    continue

            edge_pass_rate = edge_passed / edge_total if edge_total > 0 else 0.0
            print(f"{model},{edge_passed},{edge_total},{edge_pass_rate:.4f}")


if __name__ == "__main__":
    models = [
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
    model_basenames = ["gpt_4.1", "gpt_4.1_mini", "claude_3.7_sonnet", "deepseek_v3_241226", "deepseek_v3_250324",
                       "gemini_2.5_flash_250520", "qwen2.5_3b", "qwen2.5_7b", "qwen2.5_32b", "qwen3_4b", "qwen3_8b", "qwen3_32b",
                       "o4_mini", "qwq_32b", "qwen3_4b_thinking", "qwen3_8b_thinking", "qwen3_32b_thinking", "deepseek_r1",
                       "deepseek_coder_v2_16b", "qwen2.5_coder_3b", "qwen2.5_coder_7b", "qwen2.5_coder_32b",
                       "codellama_7b", "geocode_gpt"]

    task_types = ["atomic", "combined", "theme"]
    process_models(models, task_types)
    calculate_all_models_comprehensive_results(model_basenames, task_types)

    print(1)
