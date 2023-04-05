import pandas as pd
import numpy as np
import os
from tqdm import tqdm

results = {}
# list all directories
for dir in tqdm(os.listdir('/home/simon/deep-casas/ftw_model/result_new1'), total=len(os.listdir('/home/simon/deep-casas/ftw_model/result_new1'))):
    # check is directory and not equal to ipynb_checkpoints
    is_complete_exp = True
    if os.path.isdir('/home/simon/deep-casas/ftw_model/result_new1/' + dir) and dir != '.ipynb_checkpoints':
        # list all files in directory
        results[dir] = {}
        for i in range(1, 4):
            results[dir][i] = {}
            results_path = '/home/simon/deep-casas/ftw_model/result_new1/' + dir + '/' + str(i) + '/report1.txt'
            # check if file exists
            if not os.path.exists(results_path):
                is_complete_exp = False
                break

            results_df = pd.read_csv(results_path, sep='\t')
            results[dir][i]['precision'] = results_df['micro/precision']
            results[dir][i]['recall'] = results_df['micro/recall']
            results[dir][i]['f1'] = results_df['micro/f1']

            times_path = '/home/simon/deep-casas/ftw_model/result_new1/' + dir + '/' + str(i) + '/time.txt'
            results[dir][i]['times'] = float(open(times_path, 'r').read().strip())

        if not is_complete_exp:
            del results[dir]
            continue
        results[dir]['average'] = {}
        results[dir]['average']['precision'] = np.mean([results[dir][i]['precision'] for i in range(1, 4)])
        results[dir]['average']['recall'] = np.mean([results[dir][i]['recall'] for i in range(1, 4)])
        results[dir]['average']['f1'] = np.mean([results[dir][i]['f1'] for i in range(1, 4)])
        results[dir]['average']['times'] = np.mean([results[dir][i]['times'] for i in range(1, 4)])

        for i in range(1, 4):
            del results[dir][i]

        results[dir] = results[dir]['average']
        # del results[dir]['average']

results_df = pd.DataFrame(results).round(4).T
results_df.to_csv('/home/simon/deep-casas/ftw_model/result_new1/combined_results.tsv', sep='\t')
