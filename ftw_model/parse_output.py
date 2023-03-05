import numpy as np
import pandas as pd
import argparse

activitiy_mapping = {'Sleep': 0,
    'Bed_Toilet_Transition': 1,
    'Toilet': 2,
    'Take_Medicine': 3,
    'Dress': 4,
    'Work': 5,
    'Cook': 6,
    'Eat': 7,
    'Wash_Dishes': 8,
    'Relax': 9,
    'Personal_Hygiene': 10,
    'Bathe': 11,
    'Groom': 12,
    'Drink': 13,
    'Leave_Home': 14,
    'Enter_Home': 15,
    'Phone': 16,
    'Other_Activity': 17}

def parsing(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    results = []
    cache = None
    for line in lines:
        line = line.strip()
        if (line in activitiy_mapping):
            if cache:
                results.append(cache)
            cache = {'activity': line}
        elif (line[:5] == 'epoch'):
            continue
        elif (line[:3] == "1.0"):
            _, p, r, f1, support = line.split()
            cache['positive_P'] = float(p)
            cache['positive_R'] = float(r)
            cache['positive_f1'] = float(f1)
            cache['positive_support'] = int(support)
        elif (len(line.split()) == 0):
            continue
        elif (line.split()[0] in ['accuracy']):
            cache['accuracy'] = float(line.split()[1])
        elif (line.split()[0] in ['macro']):
            _, _, p, r, f1, support = line.split()
            cache['marco_p'] = float(p)
            cache['marco_r'] = float(r)
            cache['marco_f1'] = float(f1)
    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--file', dest='file', action='store', default='', help='deep model')
    args = p.parse_args()

    result = pd.DataFrame.from_dict(parsing(args.file))

    mean_series = result.mean(axis=0, numeric_only=True)
    mean_series['activity'] = 'Mean'
    print(mean_series)
    result = result.append(mean_series, ignore_index=True)
    result = result.round(4)
    result.to_csv(args.file + '.tsv', sep='\t')