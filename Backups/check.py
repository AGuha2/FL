import pandas as pd

for i in range(1,6):

    df = pd.read_csv(
        rf"D:\CAPSTONE\Dir_Client{i}.csv"
    )

    print("\nCLIENT", i)

    print(
        df['Label']
        .value_counts(normalize=True)
        .head(10)
    )