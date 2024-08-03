import time
import numpy as np
import concurrent.futures
from draft_functions import simulate_and_record
from related_functions import save_to_csv


def run_simulations(repeats):  # [1, 2, 3, 4]   # "Band1", "Band2", "CS(0.2)", "CS(0.4)", "AR(0.3)", "AR(0.7)"
    combinations = [(B_type, Correlation_type) for B_type in [1, 2, 3]
                    for Correlation_type in ["Band1"]]
    tasks = [(B_type, Correlation_type, repeat_id) for B_type, Correlation_type in combinations
             for repeat_id in range(repeats)]
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
                        'proposed': {'TPR': [], 'FPR': [], 'SSE': [], 'Cindex': [], 'RI': [], 'ARI': [], 'G': []},
                        'heter': {'TPR': [], 'FPR': [], 'SSE': [], 'Cindex': [], 'RI': [], 'ARI': [], 'G': []},
                        'homo': {'TPR': [], 'FPR': [], 'SSE': [], 'Cindex': [], 'RI': [], 'ARI': [], 'G': []},
                        'notree': {'TPR': [], 'FPR': [], 'SSE': [], 'Cindex': [], 'RI': [], 'ARI': [], 'G': []}
                    }

                for method in ['proposed', 'heter', 'homo', 'notree']:
                    for metric in data[method]:
                        results[(B_type, Correlation_type)][method][metric].append(data[method][metric])

                save_to_csv(results, filename=f"results_B{B_type}_{Correlation_type}.csv")  # 保存结果

            except Exception as exc:
                print(f"{task} generated an exception: {exc}")

    return results


if __name__ == "__main__":
    start_time = time.time()

    repeats = 1
    results = run_simulations(repeats=repeats)
    print(results)

    # 计算运行时间
    running_time = time.time() - start_time
    print(f"Running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
