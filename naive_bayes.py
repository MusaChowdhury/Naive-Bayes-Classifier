# Written By Musa Chowdhury
# Date : 8.5.2021 (dd/mm/yy)
# Python : 3.10
# Dependency : Pandas, Numpy

import pandas as pd
import numpy as np
import math

file_name = "heart_disease.csv"
central_index = {"categorical": {}, "continuous": {}, "target": None}
probability_index = {"values": []}
categorical_variable_list = None
continuous_variable_list = None
target_variable = None


def print_index():
    print("*" * 50)
    for i in central_index["target"]["values"]:
        print(f"\nWhen target is {i} (Total Example = {central_index['target'][f'target={i}']})")
        for j in central_index["categorical"].keys():
            print("     ", j)
            for z in central_index["categorical"][j]["values"]:
                try:
                    print(
                        f"            When value is {z}, frequency is {central_index['categorical'][j][f'target={i}'][f'value={z}']}")
                except:
                    print(f"            When value is {z}, frequency is ZERO",
                          ("-" * 20) + ">" " Zero Probability Detected", )

        for j in central_index["continuous"].keys():
            print("     ", j)
            print(
                f"            Mean = {central_index['continuous'][j][f'target={i}']['mean']}, Variance =  {central_index['continuous'][j][f'target={i}']['variance']}")

    print("\n", "*" * 50)


def pre_calculate_variance_mean(current_target, data_frame):
    for column_name in central_index["continuous"].keys():
        if column_name not in central_index["continuous"]:
            central_index["continuous"] = {column_name: {}}
        mean = float(sum(float(x) for x in data_frame[column_name])) / float(len(data_frame[column_name]))
        sum_x = float(0)
        for i in data_frame[column_name]:
            _x = float(i) - mean
            sum_x += _x * _x
        try:
            central_index["continuous"][column_name].update(
                {f"target={current_target}": {"mean": mean, "variance": (sum_x / (len(data_frame[column_name]) - 1))}})
        except:
            # print(column_name, target,mean,sum_x,len(values))
            central_index["continuous"][column_name].update(
                {f"target={current_target}": {"mean": -1, "variance": -1}})


def post_probability_density_estimation(column_name, target, value):
    mean = central_index["continuous"][column_name][f"target={target}"]["mean"]
    variance = central_index["continuous"][column_name][f"target={target}"]["variance"]
    temp = (1 / math.sqrt(2 * 3.1416 * variance)) * (math.exp(-1 * math.pow((value - mean), 2) / (2 * variance)))
    # print(mean, variance, temp)
    return temp


def pre_select_column_type(columns):
    for i in columns:
        match = i[-2:]
        if match == "_C":
            central_index["categorical"].update({i: {}})
        elif match == "_T":
            central_index["target"] = {"name": i}
        else:
            central_index["continuous"].update({i: {}})


def pre_frequency_calculator(current_target, data_frame):
    for i in central_index["categorical"].keys():
        values, frequency = np.unique(data_frame[i], return_counts=True)
        values = list(values)
        frequency = list(frequency)
        if "values" not in central_index["categorical"][i]:
            central_index["categorical"][i].update({"values": set()})
        central_index["categorical"][i]["values"].update(values)
        for j in values:  # {f"values={j}":frequency[values.index(j)]}
            if f"target={current_target}" not in central_index["categorical"][i].keys():
                central_index["categorical"][i].update({f"target={current_target}": {}})

            central_index["categorical"][i][f"target={current_target}"].update(
                {f"value={j}": frequency[values.index(j)]})

        # print(values, frequency, i,current_target)


def central_index_and_test_dataset_builder(file, train_dataset_percent=60):
    # Preprocess
    data_frame = pd.read_csv(file_name)
    data_frame.replace('', np.nan)
    data_frame.dropna(axis="index", how="any", inplace=True)
    # Preprocess

    train = data_frame.sample(frac=train_dataset_percent / 100, random_state=200)  # random state is a seed value
    test = data_frame.drop(train.index)
    data_frame = train

    # Sorting Column According to Type
    pre_select_column_type(data_frame.columns)
    central_index.update({"total_examples": data_frame.shape[0]})
    # Sorting Column According to Type

    # Frequency of Target Variable
    values, frequency = np.unique(data_frame[central_index["target"]["name"]].values, return_counts=True)
    values = list(values)
    frequency = list(frequency)
    central_index["target"].update({"values": values})
    for i in values:
        central_index["target"].update({f"target={i}": frequency[values.index(i)]})
    # Frequency of Target Variable

    # Splitting Dataset According to Target Variable Values
    splitted_data_frame = data_frame.groupby(data_frame[central_index["target"]["name"]])
    # Splitting Dataset According to Target Variable Values

    for i in central_index["target"]["values"]:
        current_frame = splitted_data_frame.get_group(i)
        current_target = current_frame[central_index['target']['name']].values[0]
        pre_frequency_calculator(current_target, current_frame)
        pre_calculate_variance_mean(current_target, current_frame)
    return test


def probability_index_builder():
    for i in central_index["target"]["values"]:
        probability_index["values"].append(i)
        current_class = probability_index[i] = {}
        current_class_frequency = central_index['target'][f'target={i}']
        current_class["frequency"] = central_index['target'][f'target={i}']

        for j in central_index["categorical"].keys():
            current_variable = current_class[j] = {}
            current_variable["values"] = []
            for z in central_index["categorical"][j]["values"]:
                current_variable["values"].append(z)
                try:
                    current_variable.update(
                        {z: central_index['categorical'][j][f'target={i}'][f'value={z}'] / current_class_frequency})

                except:
                    current_variable.update({z: 0})

        for j in central_index["continuous"].keys():
            current_class.update({j: {"mean": central_index['continuous'][j][f'target={i}']['mean'],
                                      "variance": central_index['continuous'][j][f'target={i}']['variance']}})


def initialize_and_training_data_builder(file, train_percent):
    test_dataset = central_index_and_test_dataset_builder(file, train_dataset_percent=train_percent)
    probability_index_builder()
    global categorical_variable_list
    global continuous_variable_list
    global target_variable
    categorical_variable_list = list(central_index["categorical"].keys())
    continuous_variable_list = list(central_index["continuous"].keys())
    target_variable = central_index["target"]["name"]
    training_data_ = []
    all_columns = list(test_dataset.columns)
    for i in test_dataset.values:
        temp = {}
        if len(i) == len(all_columns):
            for z in range(len(all_columns)):
                temp[all_columns[z]] = i[z]
            training_data_.append(temp)
    return training_data_


training_data = initialize_and_training_data_builder(file_name, 90)

accuracy = 0

for i in training_data:
    hold = {}
    hold2 = []
    print("\n" + str(i))

    for j in central_index["target"]["values"]:
        temp = 1
        for k in i:
            if k in categorical_variable_list:
                # print(j,k,i[k],probability_index[j][k][i[k]])
                if probability_index[j][k][i[k]] != 0:
                    temp *= probability_index[j][k][i[k]]
                else:
                    # Implement Zero Handler
                    pass
            elif k in continuous_variable_list:
                # print(j,k, i[k], post_probability_density_estimation(k,j,i[k]))
                if probability_index[j][k]["mean"] != -1:
                    temp *= post_probability_density_estimation(k, j, i[k])
                else:
                    # Implement Zero Handler
                    pass
        final_all_possibility = {j: temp}
        hold.update(final_all_possibility)
        hold2.append(temp)
        # print(j, temp)
    maximum_probability = max(hold2)
    for p in hold:
        if maximum_probability == hold[p]:
            print("Predicted", p, "Where Actual Value is", i[central_index["target"]["name"]], "\n")
            if p == i[central_index["target"]["name"]]:
                accuracy += 1

print((accuracy / len(training_data)) * 100)
