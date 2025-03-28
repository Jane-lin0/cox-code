import json
import time
from concurrent.futures import ProcessPoolExecutor
import sys

from Empirical.renrendai.process_function import compute_single_trial


def main():
    sys.setrecursionlimit(1 << 25)
    region_list = ["北京", "河北", "山东", "江苏", "上海", "浙江", "福建", "广东"]
        # ['北京', '河北', '山东', '江苏', '上海', '浙江', '福建', '广东',
        #            '山西', '河南', '安徽', '湖北', '湖南', '江西',
        #            '陕西', '四川', '云南', '广西',
        #            '辽宁', '吉林', '黑龙江']
    tree_structure = "G8"

    method = 'notree'
    repeats = 10
    test_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = {}
    for test_rate in test_rates:
        results[test_rate] = []

    with ProcessPoolExecutor() as executor:
        future_to_key = {}

        for test_rate in test_rates:
            for repeat_id in range(repeats):
                key = (test_rate, repeat_id)
                future = executor.submit(
                    compute_single_trial,
                    region_list,
                    test_rate,
                    repeat_id,
                    tree_structure,
                    method
                )
                future_to_key[key] = future

        for key, future in future_to_key.items():
            try:
                test_rate, repeat_id, avg_cindex, label, B_best = future.result()
                results[test_rate].append({
                    'repeat_id': repeat_id,
                    'avg_cindex': avg_cindex,
                    'label': label,
                    'B_best': B_best
                })
            except Exception as e:
                print(f"Key {key} generated an exception: {e}", file=sys.stderr)

        with open(f'{method}_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    start_time = time.time()
    results = main()

    import pandas as pd

    # 初始化Excel writer
    excel_path = 'results.xlsx'
    writer = pd.ExcelWriter(excel_path)

    # 遍历每个test_rate并生成对应的Sheet
    for test_rate in results:
        entries = results[test_rate]
        rows = []

        for entry in entries:
            row_data = {
                'test_rate': test_rate,
                'repeat_id': entry['repeat_id'],
                'avg_cindex': entry['avg_cindex']
            }
            # 展开label列表到8列
            for i in range(8):
                row_data[f'label_{i}'] = entry['label'][i]

            # 展开B_best列表到8列
            for i in range(8):
                row_data[f'B_best_{i}'] = entry['B_best'][i]

            rows.append(row_data)

        # 创建DataFrame并写入当前test_rate的Sheet
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=f'Test Rate {test_rate}', index=False)

    # 保存Excel文件
    writer.save()

    running_time = time.time() - start_time
    print(f"\nRunning time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")





# import time
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
#
# from Empirical.renrendai.process_function import process_repeat, optimize_test_rate
#
# if __name__ == "__main__":
#     start_time = time.time()
#
#     region_list = ["辽宁", "黑龙江",
#                    "湖北", "湖南",
#                    "天津", "福建", "海南",
#                    "陕西", "甘肃", "新疆"]
#
#     tree_structure = "G10"
#     repeats = 10
#     test_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]
#
#     # 并行处理不同测试率
#     all_results = []
#     with ProcessPoolExecutor() as executor:
#         rate_workers = []
#         for test_rate in test_rate_list:
#             rate_worker = partial(optimize_test_rate, test_rate, region_list, tree_structure)
#             rate_future = executor.submit(rate_worker)
#             rate_workers.append((test_rate, rate_future))
#
#         # 收集所有结果
#         for test_rate, future in rate_workers:
#             all_results.append((test_rate, future.result()))
#
#     # 整理最终结果
#     final_results = {}
#     for test_rate, res in all_results:
#         final_results[test_rate] = res
#
#     running_time = time.time() - start_time
#     print(f"\n Running time: {running_time / 60:.2f} minutes ({running_time / 3600:.2f} hours)")
