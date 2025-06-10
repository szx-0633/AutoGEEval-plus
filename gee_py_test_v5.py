import os
import re
import numpy as np
import yaml
import ee
import geemap
import json
import requests
import rasterio
import shutil
import time
import types
from shapely.geometry import shape
from contextlib import contextmanager
import threading
import _thread
import random

GEE_PROJECT_NAME = "ee-szx"
DEBUG_MODE = False

def download_file(url, destination, max_retries=1, retry_delay=30):
    """
    Download file with retry mechanism
    :param url: Download URL
    :param destination: Target file path
    :param max_retries: Maximum number of retries
    :param retry_delay: Retry delay time (seconds)
    :return: (Success flag, error message or None)
    """
    for retry in range(max_retries + 1):
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            response = requests.get(url, stream=True, timeout=100)
            response.raise_for_status()  # Raises an exception if status code is not 200

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True, None
        except Exception as e:
            error_str = str(e)

            # Check if it's a network error
            network_error = any(err in error_str for err in [
                "HTTPSConnectionPool",
                "500 Server Error",
                "Internal Server Error",
                "Client Error",
                "Bad Request for url"
                "Timeout"
            ])

            if network_error and retry < max_retries:
                wait_time = retry_delay
                print(
                    f"⚠️ Network error downloading file ({retry + 1}/{max_retries}), waiting {wait_time} seconds to retry...")
                time.sleep(wait_time)
                continue
            else:
                if retry == max_retries:
                    return False, f"Max retries exceeded. Last error: {error_str}"
                else:
                    return False, f"Error downloading file: {error_str}"


def python_constructor(loader, node):
    """
    Custom constructor for parsing Python code in !python tags.
    """
    python_code = loader.construct_scalar(node)  # Get code content
    local_vars = {}  # Define local variable space
    global_vars = globals().copy()  # Copy global variables
    try:
        # Dynamically execute code
        exec(compile(python_code, "<yaml>", "exec"), global_vars, local_vars)
        # Check if there's a function return
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, types.FunctionType):
                return var_value  # Return function object
        return None  # Return None if no function
    except Exception as e:
        print(f"Error executing Python code: {e}")
        return None


def get_params_data(params_data, max_retries=3, retry_delay=5):
    """
    Process parameter data, execute any Python functions.

    :param params_data: Parameter data dictionary
    :param max_retries: Maximum number of retries
    :param retry_delay: Retry interval time (seconds)
    :return: Processed parameter dictionary
    """
    processed_params = {}

    # Define network-related errors that need retry
    NETWORK_ERRORS = [
        "Connection aborted",
        "HTTPSConnectionPool",
        "500 Server Error",
        "Internal Server Error",
        "Client Error",
        "Bad Request for url",
        "Timeout"
    ]

    def is_network_error(error):
        """Check if it's a network-related error"""
        error_str = str(error)
        return any(err in error_str for err in NETWORK_ERRORS)

    for key, value in params_data.items():
        if callable(value):  # If value is a function type
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    result = value()  # Execute function
                    if isinstance(result, str):
                        result = result.replace("<ee-project>", GEE_PROJECT_NAME)
                    processed_params[key] = result
                    break  # Successfully executed, exit retry loop

                except Exception as e:
                    if is_network_error(e) and retry_count < max_retries:
                        retry_count += 1
                        print(f"⚠️ Network error while processing parameter '{key}' "
                              f"(attempt {retry_count}/{max_retries}): {str(e)}")
                        print(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # If not a network error or max retries reached
                        if is_network_error(e):
                            error_msg = f"Max retries ({max_retries}) exceeded for network error"
                        else:
                            error_msg = str(e)

                        print(f"❌ Error executing function for key '{key}': {error_msg}")
                        print("Error code:", f"{value}")
                        processed_params[key] = None
                        break
        else:
            # Process non-function type values
            if isinstance(value, str):
                value = value.replace("<ee-project>", GEE_PROJECT_NAME)
            processed_params[key] = value

    return processed_params


def get_download_url_with_retry(result, region, max_retries=1, retry_delay=30):
    """
    Get download URL with retry mechanism

    :param result: Earth Engine result object
    :param region: Region geometry object
    :param max_retries: Maximum number of retries
    :param retry_delay: Retry delay time (seconds)
    :return: (Success flag, URL or error message)
    """
    for retry in range(max_retries + 1):
        try:
            result_url = result.getDownloadURL({
                'region': region,
                'crs': 'EPSG:4326',
                'scale': 30,
                'format': 'GeoTIFF'
            })
            return True, result_url
        except Exception as e:
            error_str = str(e)

            # Handle special case of region too large
            if ("Total request size" in error_str) or ("Pixel grid dimensions" in error_str):
                try:
                    centroid = region.centroid(maxError=1)
                    smaller_region = centroid.buffer(1000).bounds()
                    smaller_result = result.clip(smaller_region)
                    result_url = smaller_result.getDownloadURL({
                        'region': smaller_region,
                        'crs': 'EPSG:4326',
                        'scale': 30,
                        'format': 'GeoTIFF'
                    })
                    return True, result_url
                except Exception as inner_e:
                    error_str = f"Error after region reduction: {str(inner_e)}"

            # Handle network errors
            network_error = any(err in error_str for err in [
                "HTTPSConnectionPool",
                "500 Server Error",
                "Internal Server Error",
                "Connection refused",
                "Connection reset",
                "Timeout",
                "Too Many Requests"
            ])

            if network_error and retry < max_retries:
                wait_time = retry_delay
                print(
                    f"⚠️ Network error getting download URL ({retry + 1}/{max_retries}), waiting {wait_time} seconds to retry...")
                time.sleep(wait_time)
                continue
            else:
                if retry == max_retries:
                    return False, f"Max retries exceeded. Last error: {error_str}"
                else:
                    return False, f"Error getting download URL: {error_str}"


# Add timeout handling context manager
@contextmanager
def timeout(seconds):
    """
    Timeout control context manager

    :param seconds: Timeout duration (seconds)
    """
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    finally:
        timer.cancel()


def test_single_file(file_path, config, output_directory, ref_answer_dir, max_retries=2, retry_delay=60, timeout_seconds=300):
    """
    Test a single code file. If you need to run this function separately, you need to read the yaml configuration file in advance

    :param file_path: Code file path
    :param config: Test configuration
    :param output_directory: Output directory
    :param ref_answer_dir: Reference answer directory
    :param max_retries: Maximum number of retries for network errors
    :param retry_delay: Delay time between retries (seconds)
    :param timeout_seconds: Timeout for each test case (seconds)
    :return: File test result statistics
    """

    file_name = os.path.basename(file_path)
    print(f"\n{'=' * 50}")
    print(f"Testing file: {file_name}")
    print(f"{'=' * 50}")

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # Add initialization code
    init_code = (f"import ee\nimport geemap\n"
                 f"geemap.set_proxy(port=7890)\nee.Initialize(project='{GEE_PROJECT_NAME}')\n")
    code_added = init_code + code
    # Remove malicious code lines that cause the test framework to crash: ee.data = None, setattr(data, ...
    if "ee.data = None" in code_added:
        print("⚠️ Warning: 'ee.data = None' found in the code, removing it.")
        code_added = re.sub(r"ee\.data\s*=\s*None", "", code_added)
    if "setattr(data," in code_added:
        print("⚠️ Warning: 'setattr(data' found in the code, removing it.")
        code_added = re.sub(r"setattr\(data,.*?\)", "", code_added)
    if "ee.Initialize" in code_added and "baseurl" in code_added:
        print("⚠️ Warning: 'ee.Initialize(url=baseurl)' found in the code, removing it.")
        code_added = re.sub(r"ee\.Initialize\(url=baseurl\)", "", code_added)
    if "ee.data.setCloudApiBaseUrl(baseurl)" in code_added:
        print("⚠️ Warning: 'ee.data.setCloudApiBaseUrl(baseurl)' found in the code, removing it.")
        code_added = re.sub(r"ee\.data\.setCloudApiBaseUrl\(baseurl\)", "", code_added)

    # File-level test statistics
    file_stats = {
        "file_name": file_name,
        "function_name": None,
        "total_test_cases": 0,
        "passed_test_cases": 0,
        "failed_test_cases": 0,
        "skipped_test_cases": 0,
        "file_errors": [],  # File-level errors
        "test_cases": [],  # Detailed results for each test case
        "status": "skipped"  # Initial status is skipped
    }

    # Try to parse and execute file code
    file_execution_success = False
    local_vars = {}
    function_name = None

    # Create test assets for asset tasks
    if "asset" in file_name.lower():
        try:
            init_test_assets()
        except Exception as e:
            error_str = str(e)
            if "API_KEY_INVALID" in error_str or "No scheme supplied" in error_str:
                ee.Reset()
                ee.Authenticate()
                ee.Initialize(project=GEE_PROJECT_NAME)
                init_test_assets()
            else:
                error_msg = f"Error initializing test assets: {str(e)}"
                file_stats["file_errors"].append(error_msg)
                print(f"❌ {error_msg}")
                return file_stats

    for retry in range(max_retries + 1):
        try:
            # Use regular expression to extract function name
            function_name_pattern = r"def (\w+)\("  # Match 'def' followed by function name and parenthesis
            function_matches = re.findall(function_name_pattern, code_added)

            if not function_matches:
                error_msg = "No function definition found in the file"
                file_stats["file_errors"].append(error_msg)
                print(error_msg)
                break

            function_name = function_matches[0]
            file_stats["function_name"] = function_name

            # Execute code
            exec(code_added, globals(), local_vars)
            file_execution_success = True

            break

        except Exception as e:
            error_str = str(e)
            network_error = any(err in error_str for err in [
                "HTTPSConnectionPool",
                "500 Server Error",
                "Internal Server Error",
                "Client Error",
                "Bad Request for url",
                "Connection aborted",
                "Timeout",
                "HttpError 400",
                "Invalid URL"
            ])

            if network_error and retry < max_retries:
                if "API_KEY_INVALID" in error_str or "No scheme supplied" in error_str:
                    ee.Reset()
                    ee.Authenticate()
                    ee.Initialize(project=GEE_PROJECT_NAME)
                print(
                    f"⚠️ Network error during code execution ({retry + 1}/{max_retries}), waiting {retry_delay} seconds to retry...")
                file_stats["file_errors"].append(
                    f"Network error during code execution (attempt {retry + 1}): {error_str}")
                time.sleep(retry_delay)
                continue
            else:
                if network_error:
                    error_msg = f"Max retries exceeded during code execution. Last error: {error_str}"
                else:
                    error_msg = f"Error executing code: {error_str}"

                print(f"❌ {error_msg}")
                file_stats["file_errors"].append(error_msg)
                break

    # If file code execution fails, skip this file
    if not file_execution_success:
        file_stats["status"] = "failed"
        print(f"❌ File execution failed, skipping test cases")
        return file_stats

    # Get all test data for the function
    test_cases = config.get(function_name, [])
    print(f"Testing function: {function_name}")

    if len(test_cases) == 0:
        error_msg = f"No test cases found for {function_name}"
        file_stats["file_errors"].append(error_msg)
        print(print(f"❌ No test cases found for {function_name}"))
        file_stats["status"] = "skipped"
        return file_stats

    file_stats["total_test_cases"] = len(test_cases)

    # Execute each test case
    for i, test_case in enumerate(test_cases):
        test_case_result = {
            "test_case_id": i + 1,
            "edge_test": test_case.get('edge_test', False),
            "out_type": "",
            "status": "skipped",
            "error": None,
            "retry_count": 0
        }

        print(f"Running test case {i + 1}/{len(test_cases)}...")

        # 测试用例重试机制
        for retry in range(max_retries + 1):
            test_case_result["retry_count"] = retry

            try:
                # Add timeout control
                with timeout(timeout_seconds):
                    params = test_case['params']
                    params_data = params.copy()
                    params_data = get_params_data(params_data)

                    # Pass the processed parameter data to the function
                    result = local_vars[function_name](**params_data)

                    # Check the result
                    converted_type = str(type(result))
                    test_case_result["out_type"] = converted_type

                    # If it's an error_message type test, it passes as long as no exception is thrown
                    if test_case['out_type'] == 'error_message':
                        print(f"✅ Test case {i + 1} passed (error case handled successfully)!")
                        test_case_result["status"] = "passed"
                        file_stats["passed_test_cases"] += 1
                        break
                    else:
                        # Other types of tests need to check the result
                        flag, message = check_result(result, test_case, ref_answer_dir, output_directory)
                        if flag == False:
                            message = str(message)
                            print(f"❌ Test case {i + 1} failed: {message}")
                            test_case_result["status"] = "failed"
                            test_case_result["error"] = message
                            file_stats["failed_test_cases"] += 1
                            break
                        else:
                            print(f"✅ Test case {i + 1} passed!")
                            test_case_result["status"] = "passed"
                            file_stats["passed_test_cases"] += 1
                            break

            except TimeoutError as te:
                error_msg = f"Test case timed out after {timeout_seconds} seconds"
                print(f"❌ Test case {i + 1} failed: {error_msg}")
                test_case_result["status"] = "failed"
                test_case_result["error"] = error_msg
                file_stats["failed_test_cases"] += 1
                break

            except Exception as e:
                error_str = str(e)
                network_error = any(err in error_str for err in [
                    "HTTPSConnectionPool",
                    "500 Server Error",
                    "Internal Server Error",
                    "Client Error",
                    "Bad Request for url",
                    "Connection aborted",
                    "Timeout",
                    "HttpError 400",
                    "Invalid URL"
                ])

                if network_error and retry < max_retries:
                    print(
                        f"⚠️ Network error in test case {i + 1} ({retry + 1}/{max_retries}), waiting {retry_delay} seconds to retry...")
                    time.sleep(retry_delay)
                    continue
                elif error_str.strip() == "Ok":
                    print(f"✅ Test case {i + 1} passed!")
                    test_case_result["status"] = "passed"
                    file_stats["passed_test_cases"] += 1
                    break
                else:
                    if DEBUG_MODE and test_case['out_type'] == 'error_message':
                        flag, message = check_result(error_str, test_case, ref_answer_dir, output_directory)
                        if flag:
                            print(f"✅ Test case {i + 1} passed with expected error!")
                            test_case_result["status"] = "passed"
                            file_stats["passed_test_cases"] += 1
                            break
                        else:
                            error_msg = f"Error message mismatch: {error_str}"
                            print(f"❌ Test case {i + 1} failed: {error_msg}")
                            test_case_result["status"] = "failed"
                            test_case_result["error"] = error_msg
                            file_stats["failed_test_cases"] += 1
                            break
                    else:
                        # Any unhandled exception counts as a failure
                        if network_error:
                            error_msg = f"Max retries exceeded. Last error: {error_str}"
                        else:
                            error_msg = f"Unhandled error in test case: {error_str}"

                    print(f"❌ Test case {i + 1} failed: {error_msg}")
                    test_case_result["status"] = "failed"
                    test_case_result["error"] = error_msg
                    file_stats["failed_test_cases"] += 1
                    break

        # If the test case was skipped
        if test_case_result["status"] == "skipped":
            file_stats["skipped_test_cases"] += 1

        # Add test case result to file statistics
        file_stats["test_cases"].append(test_case_result)

    # Delete test assets
    if "asset" in file_name.lower():
        delete_test_assets()

    # Determine the overall status of the file based on test case results
    if file_stats["failed_test_cases"] == 0 and file_stats["passed_test_cases"] > 0:
        file_stats["status"] = "passed"
        print(f"\n✅ All test cases passed for {file_name}")
    elif file_stats["passed_test_cases"] == 0:
        file_stats["status"] = "failed"
        print(f"\n❌ All test cases failed for {file_name}")
    else:
        file_stats["status"] = "partial"  # Some test cases passed
        print(
            f"\n⚠️ {file_stats['passed_test_cases']}/{file_stats['total_test_cases']} test cases passed for {file_name}")

    return file_stats


def init_test_assets(max_retries=2, retry_delay=30):
    def create_test_asset(test_folder, asset_name):
        # Create a constant image with test metadata
        image = ee.Image.constant(1).rename('constant_image').set({'test': True})
        full_path = f'{test_folder}/{asset_name}'

        task = ee.batch.Export.image.toAsset(
            image=image,
            description=asset_name,
            assetId=full_path,
            scale=1000,
            maxPixels=1e9,
            region=ee.Geometry.Rectangle([120, 30, 121, 31], 'EPSG:4326', False)
        )
        task.start()
        return full_path, task

    def wait_for_tasks(timeout=60, interval=2):
        # Wait for all Earth Engine tasks to complete
        start_time = time.time()
        while True:
            tasks = ee.data.listOperations()
            active_tasks = [t for t in tasks if t['metadata']['state'] in ['QUEUED', 'RUNNING']]
            if not active_tasks:
                return
            if time.time() - start_time > timeout:
                print("Error when creating test assets: timeout!\n")
                return
            time.sleep(interval)

    def delete_all_assets(folder_path):
        # Delete all assets in the specified folder
        try:
            assets = ee.data.listAssets(folder_path)
            for asset in assets.get('assets', []):
                ee.data.deleteAsset(asset["name"])
        except Exception as e:
            print(f"⚠️ Error deleting assets from {folder_path}: {str(e)}")

    folder_path = f"projects/{GEE_PROJECT_NAME}/assets/test-assets"
    folder_path2 = f"projects/{GEE_PROJECT_NAME}/assets/test-assets-to-list"

    for attempt in range(max_retries + 1):  # Includes first attempt and up to max_retries retries
        try:
            # Clean up old assets
            delete_all_assets(folder_path)
            delete_all_assets(folder_path2)

            # Create folders
            try:
                ee.data.createFolder(folder_path)
            except Exception as e:
                pass
            try:
                ee.data.createFolder(folder_path2)
            except Exception as e:
                pass

            # Create image assets
            for i in range(3):
                create_test_asset(folder_path, f'test_image_{i}')
            create_test_asset(folder_path2, 'test_image_3')

            # Create table asset
            table_asset_id = 'projects/ee-szx/assets/test-assets/test_table_asset'
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([-122.22599, 37.77045]), {'name': 'Point A', 'id': 1}),
                ee.Feature(ee.Geometry.Point([-118.24368, 34.05223]), {'name': 'Point B', 'id': 2}),
                ee.Feature(ee.Geometry.Point([-115.1398, 36.1699]), {'name': 'Point C', 'id': 3})
            ])
            task = ee.batch.Export.table.toAsset(
                collection=fc,
                description='test_table_export',
                assetId=table_asset_id,
            )
            task.start()

            wait_for_tasks()

            # Check if the number of assets meets the requirement
            assets = ee.data.listAssets(folder_path)
            if len(assets.get('assets', [])) >= 4:
                print("✅ All test assets created successfully.")
                return
            else:
                print("❌ Not enough assets were created. Retrying...")
                raise Exception("Not enough assets created.")

        except Exception as e:
            error_str = str(e)
            network_error = any(err in error_str for err in [
                "HTTPSConnectionPool",
                "500 Server Error",
                "Internal Server Error",
                "Client Error",
                "Bad Request for url",
                "Timeout",
                "googleapiclient.errors.HttpError"
            ])

            if (network_error or "Not enough assets created." in error_str) and attempt < max_retries:
                print(f"⚠️ Error occurred: {error_str}")
                print(f"🔄 Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"❌ Failed to initialize test assets after {attempt} attempts: {error_str}")
                return


def delete_test_assets():
    folder_path = f"projects/{GEE_PROJECT_NAME}/assets/test-assets"
    folder_path2 = f"projects/{GEE_PROJECT_NAME}/assets/test-assets-to-list"

    try:     
        assets = ee.data.listAssets(folder_path)
        for asset in assets['assets']:
            ee.data.deleteAsset(asset["name"])
        assets2 = ee.data.listAssets(folder_path2)
        for asset in assets2['assets']:
            ee.data.deleteAsset(asset["name"])
        test_folder_path = f"projects/{GEE_PROJECT_NAME}/assets/test-assets/test_folder"
        ee.data.deleteAsset(test_folder_path)
    except Exception as e:
        pass

    return


def run_code_from_txt(code_directory, yaml_path, output_directory, type, reference_directory="./test_code", max_retries=2, retry_delay=60):
    """
    Read code files from the specified directory, execute tests using YAML configuration, and save results to the output directory

    :param code_directory: Directory path containing code files
    :param yaml_path: Path to YAML configuration file
    :param output_directory: Directory path for output results
    :param reference_directory: Directory path for reference answers
    :param type: Test type, atomic, combined, theme
    :param max_retries: Maximum number of retries for network errors
    :param retry_delay: Delay time between retries (seconds)
    :return: None
    """
    # Register YAML constructor
    yaml.add_constructor('!python', python_constructor)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create output subdirectories for success and failure
    passed_dir = os.path.join(output_directory, "passed")
    failed_dir = os.path.join(output_directory, "failed")
    output_dir = os.path.join(output_directory, "output_results")
    report_dir = os.path.join(output_directory, "reports")
    ref_dir = os.path.join(reference_directory, f"{type}_code/ref_answer")

    os.makedirs(passed_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Get all .txt code files
    files = [f for f in os.listdir(code_directory) if f.endswith('.txt')]
    # Shuffle the evaluation order of files during large-scale parallel runs to avoid creating and deleting test assets simultaneously
    if not DEBUG_MODE:
        random.shuffle(files)

    # Read YAML configuration
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Global test statistics
    total_files = len(files)
    passed_files = 0
    failed_files = 0
    skipped_files = 0

    print(f"Found {total_files} files to test")

    # Test each file
    for file in files:
        file_path = os.path.join(code_directory, file)

        # Test single file
        file_stats = test_single_file(
            file_path,
            config,
            output_dir,
            ref_dir,
            max_retries,
            retry_delay
        )

        # Move file and update statistics based on test results
        if file_stats["status"] == "passed":
            passed_files += 1
            shutil.move(file_path, os.path.join(passed_dir, file))
        elif file_stats["status"] == "failed" or file_stats["status"] == "partial":
            failed_files += 1
            shutil.move(file_path, os.path.join(failed_dir, file))
        else:
            skipped_files += 1

        # Save test report
        report_path = os.path.join(report_dir, f"{file.replace('.txt', '')}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(file_stats, f, indent=2, ensure_ascii=False)

    # Generate overall test report
    summary_report = {
        "total_files": total_files,
        "passed_files": passed_files,
        "failed_files": failed_files,
        "skipped_files": skipped_files,
        "pass_rate": f"{(passed_files / total_files * 100):.2f}%" if total_files > 0 else "0%"
    }

    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Total files: {total_files}")
    print(f"Passed files: {passed_files}")
    print(f"Failed files: {failed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Pass rate: {summary_report['pass_rate']}")
    print("=" * 50)

    # Delete output results folder after testing if not in debug mode
    if not DEBUG_MODE:
        shutil.rmtree(output_dir)


def check_result(result, test_case, ref_answer_dir, output_directory):

    # Process the result based on output type
    out_type = test_case['out_type']
    expected_answer = test_case['expected_answer']
    answer_path = os.path.join(ref_answer_dir, expected_answer)
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)

    function_case_name = test_case['expected_answer'].split('.')[0]
    result_filename = f"{function_case_name}_output"
    result_path = os.path.join(output_directory, result_filename)

    # Check output type
    converted_type = str(type(result)).replace("ee_", "")

    if out_type == "ee.ComputedObject":
        # ee.computedobject.ComputedObject type is complex and difficult to judge, skip type comparison
        expected_type = None
    elif out_type == "ee.ArrayImage" or out_type == "ee.Image":
        # ee.ArrayImage doesn't exist in GEE, only used for result checking
        expected_type = "<class 'ee.image.Image'>"
    elif out_type.startswith("ee."):
        ee_type = out_type.split(".")[-1]
        ee_type_lower = ee_type.lower()
        expected_type = f"<class 'ee.{ee_type_lower}.{ee_type}'>"
    elif out_type == "error_message":
        expected_type = "<class 'str'>"
    else:
        expected_type = f"<class '{out_type}'>"

    if expected_type is None or converted_type == "<class 'ee.computedobject.ComputedObject'>":
        print(f"Output type is ee.ComputedObject, skipping type check.")
    else:
        if converted_type == expected_type:
            pass
        elif converted_type == "<class 'int'>" and expected_type == "<class 'float'>":
            pass
        else:
            print(f"Output type mismatch: expected {expected_type}, got {converted_type}")
            return False, f"Output type mismatch: expected {expected_type}, got {converted_type}"

    # ONLY USE IT WHEN DEBUGGING
    if DEBUG_MODE:
        result_path = answer_path

    message = None

    if out_type == "ee.Image":
        # Process image type results: compare if the image arrays are correct
        try:
            region = result.geometry()
            if (not region.getInfo()) or (region.isUnbounded().getInfo()):
                region = ee.Geometry.BBox(-1, -1, 1, 1)
                result = result.clip(region)
        except Exception:
            region = ee.Geometry.BBox(-1, -1, 1, 1)
            result = result.clip(region)

        info = result.getInfo()
        bands = len(info['bands'])

        if bands == 0:
            print("Warning: Empty image result")
            result_array = np.array([-1])
        else:
            # Get the download URL of the image
            get_flag, response = get_download_url_with_retry(result, region)
            if not get_flag:
                return False, response
            result_url = response

            # Download GeoTIFF file
            temp_tif = os.path.join(output_directory, f"temp_image.tif")
            download_flag , error = download_file(result_url, temp_tif)
            if not download_flag:
                return False, error

            # Read raster data and convert to NumPy array
            result_raster = rasterio.open(temp_tif).read()
            result_numpy = np.array(result_raster).transpose(1, 2, 0)
            result_array = np.round(result_numpy, 3)

            # Delete temporary TIF file
            if os.path.exists(temp_tif):
                os.remove(temp_tif)

        # Save as NPY file
        if not result_path.endswith('.npy'):
            result_path = f"{result_path}.npy"
        if not answer_path.endswith('.npy'):
            answer_path = f"{answer_path}.npy"
        np.save(result_path, result_array)

        # Answer check
        answer_array = np.load(answer_path, allow_pickle=True)
        # print("Expected:", answer_array)
        # print("Got:", result_array)
        try :
            np.testing.assert_array_almost_equal(result_array, answer_array, decimal=3)
            flag = True
        except AssertionError as e:
            flag = False
            message = e
        except Exception as e:
            flag = False
            message = e

    elif out_type == "ee.ArrayImage": # This type doesn't exist in GEE, only used for result checking
        # Ensure file extension is .npy
        if not result_path.endswith('.npy'):
            result_path = f"{result_path}.npy"
        if not answer_path.endswith('.npy'):
            answer_path = f"{answer_path}.npy"

        point = ee.Geometry.Point([120.05, 30.05])
        result_array0 = result.sample(point, 500).first().get('array').getInfo()
        result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('U').getInfo()
            result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('Q').getInfo()
            result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('L').getInfo()
            result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('identity').getInfo()
            result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('constant').getInfo()
            result_array = np.array(result_array0)

        # Save as NPY file
        np.save(result_path, result_array)

        # Answer check
        answer_array = np.load(answer_path, allow_pickle=True)
        # print("Expected:", answer_array)
        # print("Got:", result_array)
        try:
            np.testing.assert_array_almost_equal(result_array, answer_array, decimal=3)
            flag = True
        except AssertionError as e:
            flag = False
            message = e
        except Exception as e:
            flag = False
            message = e

    elif out_type == "ee.ImageCollection":
        # Check if the image collection is empty
        result_size = result.size().getInfo()
        if result_size == 0:
            print("Warning: Empty ImageCollection result")
            result_array = np.array([-1])
            if not result_path.endswith('.npy'):
                result_path = f"{answer_path}.npy"
            np.save(result_path, result_array)
        else:
            # Process image collection results: compare if the mean image is correct
            result = result.mean()
            try:
                region = result.geometry()
                if (not region.getInfo()) or (region.isUnbounded().getInfo()):
                    region = ee.Geometry.BBox(-1, -1, 1, 1)
                    result = result.clip(region)
            except Exception:
                region = ee.Geometry.BBox(-1, -1, 1, 1)
                result = result.clip(region)

            get_flag, response = get_download_url_with_retry(result, region)
            if not get_flag:
                return False, response
            result_url = response

            # Download GeoTIFF file
            temp_tif = os.path.join(output_directory, f"temp_image.tif")
            download_flag, error = download_file(result_url, temp_tif)
            if not download_flag:
                return False, error

            # Read raster data and convert to NumPy array
            result_raster = rasterio.open(temp_tif).read()
            result_numpy = np.array(result_raster).transpose(1, 2, 0)
            result_array = np.round(result_numpy, 3)

            # Save as NPY file
            if not result_path.endswith('.npy'):
                result_path = f"{result_path}.npy"
            if not answer_path.endswith('.npy'):
                answer_path = f"{answer_path}.npy"
            np.save(result_path, result_array)

            # Delete temporary TIF file
            if os.path.exists(temp_tif):
                os.remove(temp_tif)

        # Answer check
        answer_array = np.load(answer_path, allow_pickle=True)
        # print("Expected:", answer_array)
        # print("Got:", result_array)
        try :
            np.testing.assert_array_almost_equal(result_array, answer_array, decimal=3)
            flag = True
        except AssertionError as e:
            flag = False
            message = e
        except Exception as e:
            flag = False
            message = e

    elif out_type == "ee.Geometry":
        # Ensure file extension is .geojson
        if not result_path.endswith('.geojson'):
            result_path = f"{result_path}.geojson"
        if not answer_path.endswith('.geojson'):
            answer_path = f"{answer_path}.geojson"

        result_info = result.getInfo()
        result_geojson = json.dumps(result_info)

        # Save as GeoJSON file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_geojson)

        # Answer check
        answer_geojson = json.load(open(answer_path))
        answer_geometry = shape(answer_geojson)
        result_geometry = shape(result_info)
        # print("Expected:", answer_geojson)
        # print("Got:", result_geojson)
        if answer_geometry.equals(result_geometry):
            flag = True
        else:
            flag = False

    elif out_type == "ee.FeatureCollection" or out_type == "ee.Feature":
        # Ensure file extension is .geojson
        if not result_path.endswith('.geojson'):
            result_path = f"{result_path}.geojson"
        if not answer_path.endswith('.geojson'):
            answer_path = f"{answer_path}.geojson"

        result_info = result.geometry().getInfo()
        result_geojson = json.dumps(result_info)

        # Save as GeoJSON file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_geojson)

        # Answer check
        answer_geojson = json.load(open(answer_path))
        answer_geometry = shape(answer_geojson)
        result_geometry = shape(result_info)
        # print("Expected:", answer_geojson)
        # print("Got:", result_geojson)
        if answer_geometry.equals(result_geometry):
            flag = True
        else:
            flag = False

    elif out_type == "ee.String":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_str = str(result.getInfo())

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_str)

        # Answer check
        answer_str = open(answer_path, encoding='utf-8').read()
        # print("Expected:", answer_str)
        # print("Got:", result_str)
        if result_str == answer_str:
            flag = True
        else:
            flag = False

    elif out_type == "ee.Number":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_num = result.getInfo()

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(str(result_num))

        # Answer check
        answer_str = open(answer_path, encoding='utf-8').read()
        answer_num = float(answer_str)
        # print("Expected:", answer_num)
        # print("Got:", result_num)
        if result_num == answer_num:
            flag = True
        else:
            flag = False

    elif (out_type == "ee.Dictionary" or out_type == "ee.Reducer" or out_type == "ee.Blob"
          or out_type == "ee.Filter" or out_type == "ee.Classifier" or out_type == "ee.ErrorMargin"
          or out_type == "ee.Clusterer" or out_type == "ee.Kernel" or out_type == "ee.Element"
          or out_type == "ee.PixelType" or out_type == "ee.Join" or out_type == "ee.Projection"):
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_dict = result.getInfo()

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(json.dumps(result_dict))

        # Answer check
        answer_dict = json.load(open(answer_path))
        # print("Expected:", answer_dict)
        # print("Got:", result_dict)

        # Ignore the 'updateTime' field, its value is related to the program runtime and is not a deterministic feature in the output
        if 'updateTime' in result_dict:
            del result_dict['updateTime']
        if 'updateTime' in answer_dict:
            del answer_dict['updateTime']

        if result_dict == answer_dict:
            flag = True
        else:
            flag = False

    elif out_type == "ee.Date":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_dict = json.loads(json.dumps(result.getInfo()))
        result_date = int(result_dict['value'])

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(str(result_date))

        # Answer check
        answer_date = int(open(answer_path, encoding='utf-8').read())
        # print("Expected:", answer_date)
        # print("Got:", result_date)
        if result_date == answer_date:
            flag = True
        else:
            flag = False

    elif out_type == "ee.DateRange":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        if result.isUnbounded().getInfo():
            result_dates = [0]
        else:
            result_dict = json.loads(str(result.getInfo()).replace("'",'"'))
            result_dates = result_dict["dates"]

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(str(result_dates))

        # Answer check
        answer_dates = json.loads(open(answer_path, encoding='utf-8').read())
        # print("Expected:", answer_dates)
        # print("Got:", result_dates)
        if result_dates == answer_dates:
            flag = True
        else:
            flag = False

    elif out_type == "ee.List":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_list = result.getInfo()

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(json.dumps(result_list))

        # Answer check
        answer_list = json.load(open(answer_path))
        # print("Expected:", answer_list)
        # print("Got:", result_list)
        if result_list == answer_list:
            flag = True
        else:
            flag = False

    elif out_type == "ee.Array" or out_type == "ee.ConfusionMatrix":
        # Ensure file extension is .npy
        if not result_path.endswith('.npy'):
            result_path = f"{result_path}.npy"
        if not answer_path.endswith('.npy'):
            answer_path = f"{answer_path}.npy"

        result_array = np.array(result.getInfo())

        # Save as NPY file
        np.save(result_path, result_array)

        # Answer check
        answer_array = np.load(answer_path, allow_pickle=True)
        # print("Expected:", answer_array)
        # print("Got:", result_array)
        try:
            np.testing.assert_array_almost_equal(result_array, answer_array, decimal=3)
            flag = True
        except AssertionError as e:
            flag = False
            message = e
        except Exception as e:
            flag = False
            message = e

    elif out_type == "int" or out_type == "float":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(str(result))

        # Answer check
        answer_num = float(open(answer_path, encoding='utf-8').read())
        # print("Expected:", answer_num)
        # print("Got:", result)
        if result == answer_num:
            flag = True
        else:
            flag = False

    elif out_type == "bool" or out_type == "str":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # Save as TXT file
        result_str = str(result)
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_str)

        # Answer check
        answer_str = open(answer_path, encoding='utf-8').read()
        # print("Expected:", answer_str)
        # print("Got:", result_str)
        if result_str == answer_str:
            flag = True
        else:
            flag = False

    elif out_type == "list":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(json.dumps(result))

        # Answer check
        answer_list = json.load(open(answer_path))
        # print("Expected:", answer_list)
        # print("Got:", result)
        if result == answer_list:
            flag = True
        else:
            flag = False

    elif out_type == "dict":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(json.dumps(result))

        # Answer check
        answer_dict = json.load(open(answer_path))
        # print("Expected:", answer_dict)
        # print("Got:", result)

        # Ignore the 'updateTime' field, its value is related to the program runtime and is not a deterministic feature in the output
        if 'updateTime' in result:
            del result['updateTime']
        if 'updateTime' in answer_dict:
            del answer_dict['updateTime']

        if result == answer_dict:
            flag = True
        else:
            flag = False

    elif out_type == "error_message":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # GEE objects may only throw errors when getinfo is called
        if "ee" in converted_type:
            try:
                result = str(result.getInfo())
            except Exception as e:
                result = str(e)

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(str(result))

        # Answer check
        answer_str = str(open(answer_path, encoding='utf-8').read()).replace("ee_szx", GEE_PROJECT_NAME)

        # print("Expected:", answer_str)
        # print("Got:", result)
        if result == answer_str:
            flag = True
        else:
            print(f"Expected: {answer_str}, Got: {result}")
            flag = False

    else:
        print(f"Unsupported output type: {out_type}")
        return False, f"Unsupported output type: {out_type}"

    return flag, "Error when checking: " + str(message)


def check_model_result(model_names):
    for model_name in model_names:
        print(f"Starting test for {model_name}...")
        run_code_from_txt(f"./generate_results/{model_name}/atomic", r"./test_code/atomic_code/atomic_test_config.yaml",
                          f"./generate_results/{model_name}/atomic_output", "atomic", reference_directory="./test_code")
        run_code_from_txt(f"./generate_results/{model_name}/combined", r"./test_code/combined_code/combined_test_config.yaml",
                          f"./generate_results/{model_name}/combined_output", "combined", reference_directory="./test_code")
        run_code_from_txt(f"./generate_results/{model_name}/theme", r"./test_code/theme_code/theme_test_config.yaml",
                          f"./generate_results/{model_name}/theme_output", "theme", reference_directory="./test_code")
        print(f"Test for {model_name} completed!\n")


if __name__ == '__main__':
    geemap.set_proxy(port=7890)
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_NAME)

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
    check_model_result(models)

    print(1)
