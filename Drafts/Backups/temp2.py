import pandas as pd

df1 = pd.read_csv(r'D:\CAPSTONE\Merged01.csv')
df2 = pd.read_csv(r'D:\CAPSTONE\Merged02.csv')
df3 = pd.read_csv(r'D:\CAPSTONE\Merged03.csv')

all_labels = sorted(pd.concat([df1['Label'], df2['Label'], df3['Label']]).unique())
print(all_labels)
print("Total classes:", len(all_labels))