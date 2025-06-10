# AutoGEEval-plus

> Automated Evaluation of Large Language Models' Ability to Generate Google Earth Engine (GEE) Geospatial Code

## 📌 Project Overview

**AutoGEEval++** is a framework designed to automatically evaluate the ability of large language models (LLMs) to generate geospatial analysis code for the Google Earth Engine (GEE) platform. It includes three types of test tasks: **unit tests**, **group tests**, and **scenario tests**.

The framework supports:
- Multi-model evaluation
- Test from three levels: unit tests("Atomic" in the source code), group tests("Combined" in the source code), and scenario tests("Theme" in the source code)
- Detailed performance metrics (e.g., pass@k)
- Resource consumption tracking (tokens, inference time, line counts) and efficiency analysis

---

## 🧩 Key Files and Folder Structure

Please organize your project directory strictly as follows:

```
AutoGEEval/
├── test_code/                     # Test instructions, reference code, configs, answers
│   ├── atomic_code/               # Atomic function testing
│   ├── compositional_code/        # Compositional function testing
│   └── scenario_code/             # Scenario-based application testing
├── generate_results0/             # Intermediate raw model responses
├── generate_results/              # Extracted code from model responses
├── generate_llm_answer.py         # Script to generate model responses
├── gee_py_test_v5.py              # Script to run GEE Python tests on generated code
├── process_test_reports.py        # Script to calculate pass@k metrics
├── resource_consumption_stats.py  # Script to analyze resource usage
├── call_language_model.py         # Core API calling logic (usually not modified)
├── model_config.yaml              # Model name and API key configuration file
└── README.md                      # This file
```

---

## 🛠️ Usage Instructions

### 0. Prerequisites

- Obtain API keys for the desired LLM providers (OpenAI, Qwen, etc.)
- Set up `model_config.yaml` by copying `model_config_example.yaml` and filling in credentials
- For GEE testing, set up a GEE-enabled Google Cloud project and configure billing

---

### 1. Generate Model Responses

#### Step-by-step:

1. In `generate_llm_answer.py`, define the models you want to test in `models_to_test`:
```python
models_to_test = [
    {'provider': 'ollama', 'name': 'qwen2.5:3b', 'name_simple': 'qwen2.5_3b'}
]
```

2. Adjust the parameters of the `run_function_completion_tests()` function:
```python
run_function_completion_tests(
    test_dir="./test_code/atomic_code/test_instructions",
    models_config=models_to_test,
    type="atomic",
    stream=False,
    system_prompt=system_prompt,
    temperature=0.2,
    max_tokens=4096,
    config_path='./model_config.yaml',
    max_workers=4,
    parallel=True,
    times=5
)
```

3. Update `models_to_clean` with the expected output folder names:
```python
models_to_clean = [
    "qwen2.5_3b_1", "qwen2.5_3b_2", "qwen2.5_3b_3", 
    "qwen2.5_3b_4", "qwen2.5_3b_5"
]
```

4. Run the script. You can duplicate and run multiple instances of `generate_llm_answer.py` to speed up the process.

---

### 2. Execute Tests

#### Step-by-step:

1. Ensure all generated responses have been processed and extracted into `generate_results/`.
   - There should be:
     - 1325 `.txt` files for atomic tests
     - 1199 `.txt` files for compositional tests
     - 88 `.txt` files for scenario tests
   - Each test type also has a `summary.yaml`.

2. Replace `GEE_PROJECT_NAME` in test scripts with your own GEE project ID.
   > ⚠️ Do **not** set `DEBUG_MODE=True` unless generating reference answers.

3. In `gee_py_test_v5.py`, set the `models` list to match the generated results:
```python
models = ["qwen2.5_3b_1", ..., "qwen2.5_3b_5"]
```

4. Optionally comment out unwanted test types in `run_code_from_txt()` if you don't need to test all three categories.

5. Run the script. Multiple copies of `gee_py_test_v5.py` can be executed in parallel for faster evaluation.

---

### 3. Analyze Results

#### Step-by-step:

1. In `process_test_reports.py`, update:
```python
models = ["qwen2.5_3b_1", ..., "qwen2.5_3b_5"]
model_basenames = ["qwen2.5_3b"]
task_types = ["atomic", "compositional", "scenario"]
```

2. Run the script to compute:
   - pass@1 for each round
   - overall pass@1, pass@3, pass@5
   - edge case success rates

3. In `resource_consumption_stats.py`, similarly set:
```python
model_basenames = ["qwen2.5_3b"]
task_types = ["atomic", "compositional", "scenario"]
```

4. Run the script to get average resource statistics per model and per task type.
   - A `resource_consumption/` folder will be created containing detailed CSV reports.

---

## ⏱️ Estimated Time Consumption

| Task Type           | Approximate Time (per model, per run) |
|---------------------|----------------------------------------|
| Atomic Function Test | ~6 hours                               |
| Compositional Test   | ~3 hours                               |
| Scenario Test        | ~0.5 hours                             |

Total runtime increases linearly with the number of models and generation rounds.

---

## ⚠️ Known Issues

1. **Qwen series via Alibaba Cloud API**: Token usage cannot be retrieved due to API limitations and will be reported as 0.
2. **OpenAI o-series models (e.g., o3, o4-mini)**: These only return final outputs without reasoning steps, but token costs for the reasoning phase are still counted.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- New test tasks
- Improvements to result processing
- Enhancements to documentation
s