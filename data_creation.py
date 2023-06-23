import pandas as pd
import numpy as np

# This file has created all the subsets of classes.

# all classes stored:

classes_group = {'classes_actueel_tl': ['tl3', 'tl4'], 'classes_actueel_k': ['k4', 'k3'],
                 'classes_actueel_bk': ['bk2', 'bk1'], 'classes_actueel_gth': ['gth1', 'gth2'],
                 'classes_actueel_h': ['h5', 'h3', 'h4'], 'classes_actueel_b': ['b3', 'b4'],
                 'classes_actueel_v': ['v3', 'v4', 'v5'], 'classes_actueel_hv': ['hv1', 'hv2'],
                 'classes_actueel_all': ['tl3', 'k4', 'bk2', 'gth1', 'k3', 'h5', 'hv1', 'h3', 'h4', 'bk1', 'b3',
                                         'gth2', 'h2', 'b4', 'v3', 'tl4', 'v4', 'v5']}

classes_individual_group = {'tl3': ['tl3'], 'k4': ['k4'], 'bk2': ['bk2'], 'gth1': ['gth1'], 'k3': ['k3'], 'h5': ['h5'], 'hv1': ['hv1'], 'h3': ['h3'], 'h4': ['h4'], 'bk1': ['bk1'], 'b3': ['b3'], 'hv2': ['hv2'], 'gth2': ['gth2'], 'b4': ['b4'], 'v3': ['v3'], 'tl4': ['tl4'], 'v4': ['v4'], 'v5': ['v5']}

answer_actueel = ['tl3 has 48 students', 'k4 has 57 students', 'bk2 has 79 students', 'gth1 has 84 students',
                  'k3 has 83 students', 'h5 has 88 students', 'hv1 has 26 students', 'h3 has 45 students',
                  'h4 has 64 students',
                  'bk1 has 88 students', 'b3 has 19 students', 'hv2 has 51 students', 'gth2 has 68 students',
                  'H2 has 1 students', 'b4 has 14 students', 'v3 has 21 students', 'tl4 has 68 students',
                  'v4 has 12 students',
                  'v5 has 13 students']

classes_actueel_all = ['tl3', 'k4', 'bk2', 'gth1', 'k3', 'h5', 'hv1', 'h3', 'h4', 'bk1', 'b3', 'gth2',
                       'b4', 'v3', 'tl4', 'v4', 'v5']

classes_actueel_tl = ['tl3', 'tl4']  # has 116 students
classes_actueel_k = ['k4', 'k3']  # has 140 students
classes_actueel_bk = ['bk2', 'bk1']  # has 168 students
classes_actueel_gth = ['gth1', 'gth2']  # has 151 students
classes_actueel_h = ['h5', 'h3', 'h4']  # has 197 students
classes_actueel_b = ['b3', 'b4']  # has 33 students
classes_actueel_v = ['v3', 'v4', 'v5']  # has 46 students
classes_actueel_hv = ['hv1', 'hv2']  # has 77 students

answer_historie = ['tl3 has 112 students', 'k4 has 58 students', 'bk2 has 253 students', 'gth1 has 216 students',
                   'k3 has 149 students', 'h5 has 90 students', 'hv1 has 95 students', 'h3 has 172 students',
                   'h4 has 152 students', 'bk1 has 252 students', 'b3 has 36 students', 'hv2 has 134 students',
                   'gth2 has 204 students', 'b4 has 14 students', 'v3 has 44 students',
                   'tl4 has 84 students', 'v4 has 32 students', 'v5 has 16 students']

classes_historie = ['tl3', 'k4', 'bk2', 'gth1', 'k3', 'h5', 'hv1', 'h3', 'h4', 'bk1', 'b3', 'hv2', 'gth2', 'b4',
                    'v3', 'tl4', 'v4', 'v5']

classes_historie_tl = ['tl3', 'tl4']  # has 200 students
classes_historie_k = ['k4', 'k3']  # has 207 students
classes_historie_bk = ['bk2', 'bk1']  # has 505 students
classes_historie_gth = ['gth1', 'gth2']  # has 420 students
classes_historie_h = ['h5', 'h3', 'h4']  # has 414 students
classes_historie_b = ['b3', 'b4']  # has 50 students
classes_historie_v = ['v3', 'v4', 'v5']  # has 92 students
classes_historie_hv = ['hv1', 'hv2']  # has 229 students

answer = []

data_historie = pd.read_excel("Datasets/data_historie_language.xlsx")
data_actueel = pd.read_excel("Datasets/data_actueel_language.xlsx")

data_historie.drop(columns=data_historie.columns[0], axis=1, inplace=True)
data_historie.drop(index=0, inplace=True)

data_actueel.drop(columns=data_actueel.columns[0], axis=1, inplace=True)
data_actueel.drop(index=0, inplace=True)

data_historie_clean = data_historie.drop(['Current', 'L-year', '2-years'], axis=1)
data_complete = pd.concat([data_actueel, data_historie_clean], axis=1)

## THIS CAN BE SMALLER
# create dataset for historie
data_level_historie = data_historie
data_level_historie = data_level_historie.drop(data_level_historie.index.to_list()[:], axis=0)
data_level_historie.drop(columns=data_level_historie.columns[0:3], axis=1, inplace=True)

# create dataset for actueel
data_level_actueel = data_actueel
data_level_actueel = data_level_actueel.drop(data_level_actueel.index.to_list()[:], axis=0)
data_level_actueel.drop(columns=data_level_actueel.columns[0], axis=1, inplace=True)

# get overall data set
data_historie_all = data_level_historie
data_actueel_all = data_level_actueel


## TILL HERE
def isnotNaN(num):
    return num == num


def isNaN(num):
    return num != num


def data_class_actueel(level, data_actueel, data_level_actueel, data_all):
    a = 0
    for index, row in data_actueel.iterrows():
        this_level = row['class']
        try:
            this_level = this_level.to_string()
        except:
            a = a + 1
        this_level = str(this_level).lower()
        if this_level != level:
            continue

        # year_of_grade is in reverse, 0 = 22-23, 1 = 21-22 and 2 = 20-21
        grade_number = 0
        counter = 0
        course_counter = 0
        grades_in_order = np.full(shape=850, fill_value=np.nan)

        for grade in range(849):
            if counter == 17:
                course_counter = course_counter + 1
                counter = 0
                grade_number = 0

            temp = row[grade + 1]

            if type(temp) is str:
                temp = float(temp.replace(",", "."))

            if type(temp) is object:
                temp = float(temp)

            if isNaN(temp):
                temp = np.nan

            grades_in_order[grade_number + 17 * course_counter] = temp

            if isnotNaN(temp):
                grade_number = grade_number + 1
            counter = counter + 1
        data_level_actueel.loc[len(data_level_actueel)] = grades_in_order
        data_all.loc[len(data_all)] = grades_in_order
    return data_all


def data_class_historie(level, data_historie, data_level_historie, data_all):
    a = 0
    for index, row in data_historie.iterrows():
        this_level_current = row['Current']
        this_level_Lyear = row['L-year']
        this_level_2years = row['2-years']

        try:
            this_level_current = this_level_current.to_string()
        except:
            a = a + 1
        try:
            this_level_Lyear = this_level_Lyear.to_string()
        except:
            a = a + 1
        try:
            this_level_2years = this_level_2years.to_string()
        except:
            a = a + 1

        this_level_current = str(this_level_current).lower()
        this_level_Lyear = str(this_level_Lyear).lower()
        this_level_2years = str(this_level_2years).lower()
        grade_number = 0
        counter = 0
        course_counter = 0
        grades_in_order = np.full(shape=150, fill_value=np.nan)

        if this_level_current == level:
            # do current
            for grade in range(0, 148, 3):

                temp = row[grade + 3]
                if type(temp) is str:
                    temp = float(temp.replace(",", "."))

                if type(temp) is object:
                    temp = float(temp)

                if isNaN(temp):
                    temp = np.nan

                grades_in_order[grade] = temp
            data_level_historie.loc[len(data_level_historie)] = grades_in_order
            data_all.loc[len(data_all)] = grades_in_order

        elif this_level_Lyear == level:
            # do last year
            for grade in range(1, 149, 3):

                temp = row[grade + 3]
                if type(temp) is str:
                    temp = float(temp.replace(",", "."))

                if type(temp) is object:
                    temp = float(temp)

                if isNaN(temp):
                    temp = np.nan

                grades_in_order[grade] = temp
            data_level_historie.loc[len(data_level_historie)] = grades_in_order
            data_all.loc[len(data_all)] = grades_in_order

        elif this_level_2years == level:
            # do 2 years ago
            for grade in range(2, 150, 3):

                temp = row[grade + 3]
                if type(temp) is str:
                    temp = float(temp.replace(",", "."))

                if type(temp) is object:
                    temp = float(temp)

                if isNaN(temp):
                    temp = np.nan

                grades_in_order[grade] = temp
            data_level_historie.loc[len(data_level_historie)] = grades_in_order
            data_all.loc[len(data_all)] = grades_in_order

        else:
            continue
    return data_all


def get_data_actueel(classes, data):
    student_counter = 0
    for level in classes:
        data_level_actueel = data_actueel
        data_level_actueel = data_level_actueel.drop(data_level_actueel.index.to_list()[:], axis=0)
        data_level_actueel.drop(columns=data_level_actueel.columns[0], axis=1, inplace=True)
        data_class_actueel(level, data_actueel, data_level_actueel, data)

        data_level_actueel = data_level_actueel.astype(float)
        data_level_actueel.dropna(how='all', axis=1, inplace=True)
        string = level + " has " + str(data_level_actueel.shape[0]) + " students"
        print(string)
        answer.append(string)
        student_counter = student_counter + data_level_actueel.shape[0]
    return student_counter, data


def get_data_historie(classes, data):
    student_counter = 0
    for level in classes:
        data_level_historie = data_historie
        data_level_historie = data_level_historie.drop(data_level_historie.index.to_list()[:], axis=0)
        data_level_historie.drop(columns=data_level_historie.columns[0:3], axis=1, inplace=True)
        data_class_historie(level, data_historie, data_level_historie, data)

        data_level_historie = data_level_historie.astype(float)
        data_level_historie.dropna(how='all', axis=1, inplace=True)
        string = level + " has " + str(data_level_historie.shape[0]) + " students"
        print(string)
        answer.append(string)
        student_counter = student_counter + data_level_historie.shape[0]
    return student_counter, data


def get_data(classes, time_period, data):
    if time_period == 0:
        student_counter, data =get_data_actueel(classes, data)
    elif time_period == 1:
        get_data_historie(classes, data)
    elif time_period == 2:
        print("under construction")
    data.dropna(how='all', axis=1, inplace=True)
    return data


student_counter = 0

# Run the code here
# Time period 0 = actueel, time period 1 = historie, time period 2 is both
time_period = 0
# get_data(classes_actueel_gth, time_period, data_actueel_all)


# for actueel data
for class_set in classes_group:

    data_level_actueel = data_actueel
    data_level_actueel = data_level_actueel.drop(data_level_actueel.index.to_list()[:], axis=0)
    data_level_actueel.drop(columns=data_level_actueel.columns[0], axis=1, inplace=True)

    data_actueel_all = data_level_actueel
    data_actueel_all = get_data(classes_group[class_set], time_period, data_actueel_all)
    droplist = []
    # drop all columns with more than 20% missing values
    #for column in data_actueel_all:
        #if data_actueel_all[column].isnull().sum() / data_actueel_all.shape[0] > 0.2:
            #droplist.append(column)

    #data_actueel_all.drop(columns=droplist, inplace=True)
    data_actueel_all.to_excel(f"{class_set}_language_mixed.xlsx")








