import pandas as pd

df1 = pd.read_csv(r'D:\CAPSTONE\Merged01.csv')
df2 = pd.read_csv(r'D:\CAPSTONE\Merged02.csv')
df3 = pd.read_csv(r'D:\CAPSTONE\Merged03.csv')
df4 = pd.read_csv(r'D:\CAPSTONE\Merged04.csv')
df5 = pd.read_csv(r'D:\CAPSTONE\Merged05.csv')

labels_1 = set(df1['Label'].unique())
labels_2 = set(df2['Label'].unique())
'''labels_3 = set(df3['Label'].unique())
labels_4 = set(df4['Label'].unique())
labels_5 = set(df5['Label'].unique())'''
all_labels = labels_1 | labels_2 

print("\n=== Label 1 ===")
for l in sorted(labels_1): print(f'  "{l}",')

print("\n=== Label 2 ===")
for l in sorted(labels_2): print(f'  "{l}",')

print("\n=== ALL labels ===")
for l in sorted(all_labels): print(f'  "{l}",')

print(f"\nTotal classes: {len(all_labels)}")
print(f"Client 1 only: {labels_1 - labels_2}")
print(f"Client 2 only: {labels_2 - labels_1}")
print(f"Shared: {labels_1 & labels_2 }")

print("=== Client 1 label distribution ===")
print(df1['Label'].value_counts())

print("\n=== Client 2 label distribution ===")
print(df2['Label'].value_counts())