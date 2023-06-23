from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import os as os

outdir = r'./Results'

# check for each course in directory how many students took each course.
# Used for manual control of the amount of students in each course. It was faster to manually sort than to create a method.
for f in listdir(r'C:\Users\justu\PycharmProjects\pythonProject\Test sets\classesfull_summised_languages_mixed_h_v'):
    path = f"Test sets\classesfull_summised_languages_mixed_h_v\\{f}"
    df = pd.read_excel(path)
    print(f"file {f} has {len(df.columns)} columns")
    df = df.filter(regex='1', axis=1)
    print(f"file {f} has {len(df.columns)} columns")

    i = df.shape[0] / 1
    percentage_full = (1 - df.isna().sum() / i) * 100

    percentage_full = percentage_full.to_frame()
    percentage_full.index = percentage_full.index.map(lambda x: x.rstrip(': G1'))

    percentage_full.index = percentage_full.index.str.title()
    percentage_full['students'] = pd.DataFrame(percentage_full[0] * i / 100)
    percentage_full = percentage_full.sort_values(by='students',ascending=False)
    percentage_full = percentage_full.round(2)
    percentage_full = percentage_full.rename(columns={percentage_full.columns[0]: "Percentage"})
    percentage_full.loc[percentage_full.shape[0]] = ['Total amount of students', i]
    percentage_full = percentage_full.round({'students': 0})
    pd.options.display.float_format = '{:,.0f}'.format
    print(percentage_full)
    current_class = f.replace("classes_actueel_", "")
    current_class = current_class.replace(".xlsx", "")
    current_class = current_class.replace("_language_mixed", "")
    current_class = current_class.title()
    percentage_full.to_excel(f"{current_class}.xlsx")
