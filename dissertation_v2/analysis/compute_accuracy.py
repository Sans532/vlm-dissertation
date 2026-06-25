"""Compute accuracy across result CSVs (simple placeholder)."""
import csv
import glob


def compute_accuracy(folder):
    rows = glob.glob(folder + '/*.csv')
    print('Found', len(rows), 'csv files in', folder)
    # Placeholder: just list filenames
    for p in rows:
        print('-', p)

if __name__ == '__main__':
    compute_accuracy('results/qwen')
