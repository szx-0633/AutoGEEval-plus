import os
import json
import yaml
from pathlib import Path
from collections import defaultdict


def process_test_reports(reports_dir, output_yaml_path, configs):
    """
    处理测试报告文件夹，合并为YAML文件并生成统计信息

    Args:
        reports_dir: 测试报告所在文件夹路径
        output_yaml_path: 输出的合并YAML文件路径
        configs: 原始的测试用例配置
    """

    def truncate_error_message(error_msg, max_chars=2000):
        """
        截断错误信息，按字符长度限制

        Args:
            error_msg: 错误信息字符串
            max_chars: 最大字符数，默认2000字符

        Returns:
            截断后的错误信息
        """
        if not isinstance(error_msg, str):
            error_msg = str(error_msg)

        if len(error_msg) <= max_chars:
            return error_msg

        # 截断并添加提示信息
        truncated_msg = error_msg[:max_chars]
        truncated_msg += f"\n\n... [错误信息已截断，原始长度{len(error_msg)}字符，仅显示前{max_chars}字符]"

        return truncated_msg

    # 存储所有测试报告的字典，以function_name为键
    all_reports = []

    # 遍历文件夹中的所有JSON文件
    total_count = 0
    for file_path in Path(reports_dir).glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                report = json.load(f)
                function_name = report["function_name"]
                file_errors = report.get("file_errors", [])

                if len(file_errors) > 0:
                    # 如果存在file_errors，则从配置文件加载该任务测试用例，全部作为失败
                    test_cases_config = configs.get(function_name, [])

                    for i, test_case in enumerate(test_cases_config):
                        # 截断错误信息
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
                        # 设置其他必要字段
                        test_case["function_name"] = function_name

                        # 如果测试用例中也有错误信息，同样进行截断
                        if "error" in test_case:
                            test_case["error"] = truncate_error_message(test_case["error"])

                        all_reports.append(test_case)

            except json.JSONDecodeError:
                print(f"无法解析JSON文件: {file_path}")
                total_count += 1
                continue

            except Exception as e:
                print(f"处理文件 {file_path} 时发生错误: {e}")
                total_count += 1
                continue

            total_count += 1

    with open(output_yaml_path, "w", encoding="utf-8") as output_file:
        yaml.dump(all_reports, output_file, allow_unicode=True)

    print(f"已合并 {total_count} 个测试报告到 {output_yaml_path}， 总测试用例数: {len(all_reports)}")


def python_constructor(loader, node):
    """
    自定义构造器，用于解析 !python 标签中的 Python 代码，在本文件中无需进行任何处理。
    """
    return None  # 如果没有函数，返回 None


def process_models(models, task_types):
    yaml.add_constructor('!python', python_constructor)

    for model in models:
        for task_type in task_types:
            reports_directory = f"./generate_results/{model}/{task_type}_output/reports"  # 测试报告所在文件夹
            merged_yaml_path = f"./generate_results/{model}/{task_type}_test_reports.yaml"  # 合并后的YAML文件路径

            with open(f"./test_code/{task_type}_code/{task_type}_test_config.yaml", 'r', encoding='utf-8') as f:
                  config = yaml.load(f, Loader=yaml.Loader)

            process_test_reports(reports_directory, merged_yaml_path, config)


def classify_error_type(test_case):
    """
    根据错误信息分类错误类型（仅用于非边缘测试用例）
    """
    if test_case.get('status') != 'failed':
        return None

    # 如果是边缘测试用例，不进行错误分类
    if test_case.get('edge_test', False):
        return None

    error_msg = test_case.get('error', '').lower()

    # 根据错误信息确定错误类型（非边缘测试用例）
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
        print(f"未分类的错误信息: {error_msg}")
        error_type = 'other error'

    return error_type


def calculate_single_run_pass_rate(model, task_type, run_index):
    """
    计算单次运行的通过率
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
        print(f"警告: 文件不存在 {yaml_file}")
        return 0.0

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            test_reports = yaml.safe_load(f) or []

        # 计算通过的测试用例数
        passed_tests = sum(1 for test_case in test_reports
                           if test_case.get('status') == 'passed')

        return passed_tests / total_tests

    except Exception as e:
        print(f"读取文件 {yaml_file} 时发生错误: {e}")
        return 0.0


def calculate_pass_at_k(model, task_type, k):
    """
    计算pass@k通过率
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

    # 存储每个测试用例的结果 (function_name, test_case_id) -> [status1, status2, ...]
    test_case_results = defaultdict(list)

    # 读取k次运行的结果
    for run_index in range(1, k + 1):
        yaml_file = f"./generate_results/{model}_{run_index}/{task_type}_test_reports.yaml"
        yaml_path = Path(yaml_file)

        if not yaml_path.exists():
            continue

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                test_reports = yaml.safe_load(f) or []

            # 处理每个测试用例的结果
            for test_case in test_reports:
                function_name = test_case.get('function_name', '')
                test_case_id = test_case.get('test_case_id', 0)
                status = test_case.get('status', 'failed')

                test_key = (function_name, test_case_id)
                test_case_results[test_key].append(status == 'passed')

        except Exception as e:
            continue

    # 计算pass@k
    passed_tests = 0
    for test_key, results in test_case_results.items():
        if len(results) > 0 and any(results):
            passed_tests += 1

    return passed_tests / total_tests if total_tests > 0 else 0.0


def calculate_non_edge_error_distribution(model, task_type):
    """
    计算非边缘测试用例的错误类型分布
    """
    error_counts = defaultdict(int)
    total_non_edge_failed = 0

    # 统计所有5次运行的错误类型（仅非边缘测试用例）
    for run_index in range(1, 6):
        yaml_file = f"./generate_results/{model}_{run_index}/{task_type}_test_reports.yaml"
        yaml_path = Path(yaml_file)

        if not yaml_path.exists():
            continue

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                test_reports = yaml.safe_load(f) or []

            for test_case in test_reports:
                # 只统计非边缘测试用例的失败情况
                if (test_case.get('status') == 'failed' and
                        not test_case.get('edge_test', False)):
                    total_non_edge_failed += 1
                    error_type = classify_error_type(test_case)
                    if error_type:
                        error_counts[error_type] += 1

        except Exception as e:
            continue

    # 计算比例
    error_proportions = {}
    if total_non_edge_failed > 0:
        for error_type, count in error_counts.items():
            error_proportions[error_type] = count / total_non_edge_failed

    return error_proportions


def calculate_edge_case_pass_rate(model, task_type):
    """
    计算边缘测试用例的通过率
    """
    edge_passed = 0
    edge_total = 0

    # 统计所有5次运行的边缘测试用例结果
    for run_index in range(1, 6):
        yaml_file = f"./generate_results/{model}_{run_index}/{task_type}_test_reports.yaml"
        yaml_path = Path(yaml_file)

        if not yaml_path.exists():
            continue

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                test_reports = yaml.safe_load(f) or []

            for test_case in test_reports:
                # 只统计边缘测试用例
                if test_case.get('edge_test', False):
                    edge_total += 1
                    if test_case.get('status') == 'passed':
                        edge_passed += 1

        except Exception as e:
            continue

    # 计算边缘测试用例通过率
    edge_pass_rate = edge_passed / edge_total if edge_total > 0 else 0.0

    return edge_pass_rate


def calculate_all_models_comprehensive_results(models, task_types):
    """
    计算所有模型的综合结果，包括通过率、非边缘错误类型分布和边缘测试用例通过率
    """
    # 定义非边缘错误类型顺序
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
        print(f"\n=== {task_type.upper()} 任务类型结果 ===")

        # 构建表头
        header = "Model,Run1,Run2,Run3,Run4,Run5,Average,Pass@1,Pass@3,Pass@5,Edge_Pass_Rate"
        for error_type in non_edge_error_types:
            header += f",{error_type}"
        print(header)

        for model in models:
            # 计算单次运行通过率
            single_runs = []
            sum_pass = 0.0
            for run_index in range(1, 6):
                single_rate = calculate_single_run_pass_rate(model, task_type, run_index)
                single_runs.append(f"{single_rate:.4f}")
                sum_pass += single_rate
            average = sum_pass / 5

            # 计算pass@k
            pass_at_1 = calculate_pass_at_k(model, task_type, 1)
            pass_at_3 = calculate_pass_at_k(model, task_type, 3)
            pass_at_5 = calculate_pass_at_k(model, task_type, 5)

            # 计算边缘测试用例通过率
            edge_pass_rate = calculate_edge_case_pass_rate(model, task_type)

            # 计算非边缘错误类型分布
            non_edge_error_distribution = calculate_non_edge_error_distribution(model, task_type)

            # 构建输出行
            result_line = f"{model},{','.join(single_runs)},{average:.4f},{pass_at_1:.4f},{pass_at_3:.4f},{pass_at_5:.4f},{edge_pass_rate:.4f}"

            # 添加非边缘错误类型比例
            for error_type in non_edge_error_types:
                proportion = non_edge_error_distribution.get(error_type, 0.0)
                result_line += f",{proportion:.4f}"

            print(result_line)


def calculate_detailed_edge_statistics(models, task_types):
    """
    计算详细的边缘测试用例统计信息
    """
    print(f"\n=== 边缘测试用例详细统计 ===")

    for task_type in task_types:
        print(f"\n--- {task_type.upper()} 边缘测试用例统计 ---")
        print("Model,Edge_Passed,Edge_Total,Edge_Pass_Rate")

        for model in models:
            edge_passed = 0
            edge_total = 0

            # 统计所有5次运行的边缘测试用例结果
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