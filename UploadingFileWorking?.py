import pandas as pd

# Here 0th column will be extracted
df = pd.read_excel("D:\Workbook-NamesIncludedUseful.xlsx",
				index_col = 0)

print(df)
