#####
#   Mikhailangelo Panzo
#   Comsci 35 - Data Mining
#
#   This python code is a Association Rule Mining Algorithm based on Apriori
#
#   1. set support and confidence
#   2. extract data from csv to 2d array
#   3. categorize 2d array to numerical indicators
#   4. create an array of dictionary containing itemset-frequency
#   5. iterate counting, purging, and combining by:
#       a. listing all possible n-itemset based on previous itemsets,
#           if there are none, end iteration
#       b. counting the occurence of itemset and assigning it to the corresponding key
#       c. deleting the itemset if it does not fall under support condition
#       d. raising n by 1, where n is 1 in first iteration
#   6. extract last 2 array, if there are 1, keep 1, and if none, output no rules
#   7. generate sets based on extracted array
#   8. generate possible rules based on set, transform them into dictionary of set-rules
#   9. calculate confidence, deleting rule if it fall under confidence condition
#   10. export rules
#####

import pandas
import csv
from sklearn.preprocessing import LabelEncoder


def extract_data():
    """Extracts data from a csv file named "raw" into a dictionary of data and header"""
    data = []  # main data to work with

    with open("raw.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            data.append(row)

    return {'data': data, 'header': header}


def set_amount(message):
    """Sets amount from 0 to 100 percent and prompts using [message] string"""

    choice = input(message + ". Enter percentage (0 - 100): ")
    while True:
        if choice.isnumeric():
            choice = int(choice)
            if 0 <= choice <= 100:
                return choice
        choice = input("Invalid input. Enter a number from 0 to 100: ")


def category_numerification(dataset):
    """translates categorical data into numeric values, passing a dictionary of reference and data"""

    categories = []
    numeric_data = []

    # assigns numerical value to a category, and saves category reference
    for row in dataset:
        data_row = [int(row[0])]
        for row_index in range(1, len(row)):
            if not row[row_index] == "":
                has_reference = False
                for category_index in range(len(categories)):
                    if categories[category_index] == row[row_index]:
                        data_row.append(category_index)
                        has_reference = True
                if not has_reference:
                    data_row.append(len(categories))
                    categories.append(row[row_index])
        numeric_data.append(data_row)

    return {"reference": categories, "data": numeric_data}


def apriori_support(transaction_data, support_count):
    """purges possible itemsets based on support count and returns an array of supported itemsets with rows of
    supported n-itemsets """

    support_count = support_count / 100

    dataset = transaction_data["data"]
    all_frequent_itemsets = []
    total_num_transaction = 0

    # first layer itemset-frequency , frequent_itemsets have itemset and frequency in list
    itemsets = []
    for category_value in range(len(transaction_data["reference"])):
        itemsets.append([[category_value], 0])
    for row in dataset:
        for item_index in range(len(row)):
            itemsets[row[item_index]][1] += row[0]
        total_num_transaction += row[0]
    frequent_itemsets = [x for x in itemsets if not x[1] / total_num_transaction < support_count]
    all_frequent_itemsets.append(frequent_itemsets)
    # print(frequent_itemsets)

    # apriori algorithm
    k = 1
    while len(frequent_itemsets) > 1:
        # candidate itemsets do not have frequencies
        generate_candidates(frequent_itemsets, k)

    return all_frequent_itemsets


def generate_candidates(itemsets, k):
    """takes a list of itemsets and generates itemsets of size k + 1"""

    new_itemsets = []

    # create a list of itemsets that are paired
    for i in range(len(itemsets) - 1):
        for j in range(i + 1, len(itemsets)):
            new_itemsets.append(itemsets[i][0] + itemsets[j][0])

    # remove duplicate elements in each itemset and itemsets that exceed size k + 1
    new_clean_itemsets = []
    for itemset in new_itemsets:
        clean_itemset = []
        [clean_itemset.append(x) for x in itemset if x not in clean_itemset]
        if len(clean_itemset) == k + 1:
            clean_itemset.append(clean_itemset)

# Main Code

# step 1: prompt user to set support
support = set_amount("Enter support amount")
confidence = set_amount("Enter confidence amount")

# step 2: extract data from csv to 2d array
transaction_table = extract_data()

# step 3: translate categories to numerical representation
processed_transaction_table = category_numerification(transaction_table['data'])

# step 4 and 5: create an array of itemset-frequency
supported_itemsets = apriori_support(processed_transaction_table, support)

