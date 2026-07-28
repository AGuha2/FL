"""Create the severe alpha=0.1 equal-size seven-class client split.

This intentionally reuses the tested capacity-constrained allocation code in
create_7class_equal_split.py. It writes new files, so the alpha=0.5 data is
not changed or overwritten.
"""
import create_7class_equal_split as splitter


splitter.ALPHA = 0.1
splitter.MIN_ROWS_PER_CLASS_PER_CLIENT = 0
splitter.SEED = 42
splitter.rng = splitter.np.random.RandomState(splitter.SEED)

splitter.OUTPUT_TEMPLATE = (
    r"D:\CAPSTONE\Dir7EqualA01_Client{client_id}.csv"
)
splitter.SUMMARY_PATH = (
    r"D:\CAPSTONE\Dir7EqualA01_Distribution.csv"
)


if __name__ == "__main__":
    splitter.main()
