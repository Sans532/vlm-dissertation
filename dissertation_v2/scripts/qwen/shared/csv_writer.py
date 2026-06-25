"""CSV writer utility (placeholder)."""
import csv

def write_results(path, rows, headers=None):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        if headers:
            w.writerow(headers)
        w.writerows(rows)

if __name__ == "__main__":
    write_results('example.csv', [[1,2,3]], ['a','b','c'])
