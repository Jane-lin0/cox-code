import time
import numpy as np
import pandas as pd
import concurrent.futures
from related_functions import generate_latex_table
from draft_functions import simulate_and_record


def run_simulations(repeats):
    combinations = [(B_type, Correlation_type) for B_type in [1, 2, 3, 4]
                    for Correlation_type in ["Band1", "Band2", "CS(0.2)", "CS(0.4)"]]   # "AR(0.3)", "AR(0.7)"
    tasks = [(B_type, Correlation_type, repeat_id) for B_type, Correlation_type in combinations for repeat_id in range(repeats)]
    results = {}

    # 使用 ProcessPoolExecutor 并行处理任务
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_task = {
            executor.submit(simulate_and_record, B_type, Correlation_type, repeat_id): (B_type, Correlation_type, repeat_id)
            for B_type, Correlation_type, repeat_id in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                (B_type, Correlation_type, repeat_id), data = result

                if (B_type, Correlation_type) not in results:
                    results[(B_type, Correlation_type)] = {
                        'no_tree': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []},
                        'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'c_index': [], 'RI': [], 'G': []}
                    }

                for method in ['no_tree', 'proposed']:
                    for metric in data[method]:
                        results[(B_type, Correlation_type)][method][metric].append(data[method][metric])

            except Exception as exc:
                print(f"{task} generated an exception: {exc}")

    # 计算每个组合的平均值和标准差
    for combination, data in results.items():
        for method in ['no_tree', 'proposed']:
            for metric in data[method]:
                mean_value = np.mean(data[method][metric])
                std_value = np.std(data[method][metric])
                results[combination][method][metric] = {'mean': mean_value, 'std': std_value}

    return results


if __name__ == "__main__":
    start_time = time.time()

    repeats = 2
    results = run_simulations(repeats=repeats)  # 或者改为100
    print(results)

    # 计算运行时间
    running_time = time.time() - start_time
    print(f"Running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
