import pandas as pd
import numpy as np
from tqdm import tqdm
import os as os
from os import listdir
from os.path import isfile, join
import warnings

warnings.filterwarnings('ignore')

# Get all the names of the students
names = []

# all permanenent variables
nogo = {'G', 'ma', 'go', 'V', 'zg', 'ui', '-', 'vo', 'rv', 'O', 'T'}
nogo_courses = ['Gem.', "R1", "R2", 'R3', 'SE']
empty_file_students = []

path_historie = r"C:\Users\justu\Pandas code\Student files\Historie"
path_actueel = r"C:\Users\justu\Pandas code\Student files\Actueel"

historie_start = "\Details - Leerling - Cijfers - Historie -"
historie_actueel = "\Details - Leerling - Cijfers - Actueel -"

historie_end = "- Cijfers.xlsx"
actueel_end = "- Cijfers.xlsx"

amount_of_courses = 150
amount_of_courses_actueel = 850


# This processes the students grades into the main database
def isNaN(num):
    return num != num
def process_student_historie(student_name, data_historie):
    file_name_actueel = path_actueel + historie_actueel + student_name + actueel_end
    file_name_historie = path_historie + historie_start + student_name + historie_end

    historie = pd.read_excel(file_name_historie, skiprows=20)
    actueel = pd.read_excel(file_name_actueel, skiprows=21)

    # Rename the first column to Course
    historie.rename(columns={historie.columns[0]: "Course"}, inplace=True)

    # To keep the data much cleaner remove all grades from the year 2019 and the unnamed.
    # Since there are less than 30 report cards with grades from 2019, all containing only a few grades, this data is best ommited.
    historie.drop(list(historie.filter(regex='19|nnamed')), axis=1, inplace=True)

    # Remap all the courses to numbers
    historie['Course'] = historie['Course'].astype(str).str.lower()
    historie = historie.replace({"Course": dictionary})

    # Change the format of the grades to be easier processable
    grades_list_format_historie = [historie.columns.tolist()] + historie.reset_index().values.tolist()

    # repeat for current grades
    if student_name not in empty_file_students:
        actueel.rename(columns={actueel.columns[0]: "Course"}, inplace=True)
        actueel['Course'] = actueel['Course'].astype(str).str.lower()
        actueel = actueel.replace({"Course": dictionary})
        grades_list_format_actueel = [actueel.columns.tolist()] + actueel.reset_index().values.tolist()

    # Copy all the grades into the correct format.
    grades_in_order = np.full(shape=amount_of_courses, fill_value=np.nan)

    # find the classes this student has been in.
    path_overzicht = r"C:\Users\justu\Pandas code\Student files\Overzicht"
    overzicht_start = "\Details - Leerling - Overzicht -"
    overzicht_end = "- Overzicht.xlsx"
    file_name_overzicht = path_overzicht + overzicht_start + student_name + overzicht_end
    classes_data = pd.read_excel(file_name_overzicht, skiprows=14)
    classes_data = pd.DataFrame(classes_data, columns=['Klas'])

    try:
        classes_data['Klas'] = [str(x).split(' ')[0] for x in classes_data['Klas'].to_list()]
        classes_data['Klas'] = [str(x)[:-1] for x in classes_data['Klas'].to_list()]
    except AttributeError as e:
        print(f"Error {e} with dataframe")
        print(classes_data['Klas'])

    while len(classes_data) < 3:
        classes_data.loc[len(classes_data)] = 'no data'

    if len(classes_data) > 3:
        classes_data = classes_data[:3]
        # Course grades
    # Check amount of stored data, but no more than 3
    amount_of_stored_years = len(historie.columns)
    if len(historie.columns) > 4:
        amount_of_stored_years = 4

    for course, rows in historie.iterrows():
        # check what course this is
        course_number = grades_list_format_historie[course][1]
        if type(course_number) is not int:
            continue
        # year_of_grade is in reverse, 0 = 22-23, 1 = 21-22 and 2 = 20-21
        for year_of_grade in range(amount_of_stored_years - 1):
            temp = grades_list_format_historie[course][amount_of_stored_years - year_of_grade]
            if temp in nogo:
                continue
            if isNaN(temp):
                continue
            if type(temp) is str:
                temp = float(temp.replace(",", "."))
            grades_in_order[course_number * 3 + year_of_grade] = temp
    classes = np.array(classes_data)
    grades_in_order = np.append(classes, grades_in_order)
    data_historie.loc[len(data_historie)] = grades_in_order


def process_student_actueel(student_name, data_actueel):
    file_name_actueel = path_actueel + historie_actueel + student_name + actueel_end

    actueel = pd.read_excel(file_name_actueel, skiprows=22)
    grades_in_order = np.full(shape=amount_of_courses_actueel, fill_value=np.nan)
    if student_name in empty_file_students:
        grades_in_order = np.full(shape=amount_of_courses_actueel + 1, fill_value=np.nan)
        data_actueel.loc[len(data_actueel)] = grades_in_order
        return

    # change the column names to only contain the last number

    # Rename the first column to Course
    # Change the format of the grades to be easier processable
    actueel.rename(columns={actueel.columns[0]: "Course"}, inplace=True)
    actueel['Course'] = actueel['Course'].astype(str).str.lower()

    # Remap all the courses to numbers
    actueel = actueel.replace({"Course": dictionary})
    grades_list_format_actueel = [actueel.columns.tolist()] + actueel.reset_index().values.tolist()

    # repeat for current grades
    # Copy all the grades into the correct format.

    # find the classes this student has been in.
    path_overzicht = r"C:\Users\justu\Pandas code\Student files\Overzicht"
    overzicht_start = "\Details - Leerling - Overzicht -"
    overzicht_end = "- Overzicht.xlsx"
    file_name_overzicht = path_overzicht + overzicht_start + student_name + overzicht_end
    classes_data = pd.read_excel(file_name_overzicht, skiprows=14)
    classes_data = pd.DataFrame(classes_data, columns=['Klas'])

    try:
        classes_data['Klas'] = [str(x).split(' ')[0] for x in classes_data['Klas'].to_list()]
        classes_data['Klas'] = [str(x)[:-1] for x in classes_data['Klas'].to_list()]
    except AttributeError as e:
        print(f"Error {e} with dataframe")
        print(classes_data['Klas'])

    if len(classes_data) > 1:
        classes_data = classes_data[:1]
        # Course grades
    # Check amount of stored data, but no more than 1
    for course in nogo_courses:
        if course in actueel.columns:
            actueel.drop([course], axis=1, inplace=True)

    amount_of_stored_years = len(actueel.columns)
    amount_grades = 0
    for course, rows in actueel.iterrows():
        # check what course this is
        course_number = grades_list_format_actueel[course][1]
        if type(course_number) is not int:
            continue
        # year_of_grade is in reverse, 0 = 22-23, 1 = 21-22 and 2 = 20-21
        grade_number = 0
        for year_of_grade in range(amount_of_stored_years - 1):
            grades_list_format_actueel[0][year_of_grade] = grades_list_format_actueel[0][year_of_grade][-1]
            if grades_list_format_actueel[0][year_of_grade].isdigit() is not True:
                continue
            temp = grades_list_format_actueel[course][year_of_grade + 1]
            if isNaN(temp):
                continue
            if temp in nogo:
                continue
            if type(temp) is str:
                temp = float(temp.replace(",", "."))
            grades_in_order[course_number * 17 + grade_number] = temp
            grade_number = grade_number + 1
    classes = np.array(classes_data)
    grades_in_order = np.append(classes, grades_in_order)
    data_actueel.loc[len(data_actueel)] = grades_in_order
    # print(student_name)
    # print(amount_grades)


def student_names(f, names):
    split_string = f.split('-')
    names.append(split_string[4])


for f in listdir(r'C:\Users\justu\Pandas code\Student files\Historie'):
    student_names(f, names)

students = []
[students.append(x) for x in names if x not in students]
len(students)

# import the dictonary used to translate the course names into numbers
dictionary = pd.read_excel("Dictionary courses - languages.xlsx")

dictionary['Courses'].str.strip()
dictionary = dictionary.to_dict()
dictionary = {v: k for k, v in dictionary['Courses'].items()}

dictionary.update({'duitse taal en literatuur':10, 'engelse taal en literatuur': 12,'franse taal en literatuur': 13,'nederlandse taal en literatuur':30})
# Create the data set
data_historie = pd.DataFrame(index=np.arange(1), columns=np.arange(amount_of_courses + 3))
data_actueel = pd.DataFrame(index=np.arange(1), columns=np.arange(amount_of_courses_actueel + 1))

keys_historie = ["Current", "L-year", "2-years"]
for key in dictionary.keys():
    keys_historie.append(f"{key}:22-23")
    keys_historie.append(f"{key}:21-22")
    keys_historie.append(f"{key}:20-21")

column_names = {k: v for k, v in zip(range(amount_of_courses + 3), keys_historie)}

data_historie.rename(columns=column_names, inplace=True)

for student_name in tqdm(students):
    process_student_historie(student_name, data_historie)

keys_actueel = ["class"]
for key in dictionary.keys():
    [keys_actueel.append(f"{key}: G{j}") for j in range(1, 18)]

column_names_actueel = {k: v for k, v in zip(range(amount_of_courses_actueel + 1), keys_actueel)}
data_actueel.rename(columns=column_names_actueel, inplace=True)

for student_name in tqdm(students):
    process_student_actueel(student_name, data_actueel)

percentage_empty = data_historie.isna().sum() / 984 * 100

print(data_historie)
