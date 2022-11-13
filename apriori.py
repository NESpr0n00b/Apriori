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
        itemsets.append([{category_value}, 0])
    for transaction in dataset:
        for item_index in range(len(transaction)):
            itemsets[transaction[item_index]][1] += transaction[0]
        total_num_transaction += transaction[0]
    frequent_itemsets = [x for x in itemsets if not x[1] / total_num_transaction < support_count]
    all_frequent_itemsets.append(frequent_itemsets)
    # print(frequent_itemsets)

    # apriori algorithm
    k = 1
    while True:
        indices_to_be_deleted = []
        # candidate itemsets do not have frequencies
        frequent_itemsets = generate_candidates(frequent_itemsets, k)
        # breaks code if no more itemsets can be made
        if len(frequent_itemsets) == 0:
            break
        # for each transaction count them in frequent itemsets
        for transaction_index in range(len(dataset)):
            transaction_items = dataset[transaction_index].copy()
            transaction_items.pop(0) # removes the column that specifies count of that itemset
            is_counted = False  # checks if transaction is counted
            for itemset in frequent_itemsets:
                if itemset[0].issubset(set(transaction_items)):
                    itemset[1] += dataset[transaction_index][0]
                    is_counted = True
            # if transaction is not counted, add it to a list of indices to be deleted after, optimization code
            if not is_counted:
                indices_to_be_deleted.append(transaction_index)
        # prunes the dataset from the indices that do not have the candidate itemsets
        for indices_index in reversed(range(len(indices_to_be_deleted))):
            dataset.pop(indices_to_be_deleted[indices_index])
        # add the itemsets to the final supported itemsets based on minimum support
        frequent_itemsets = [x for x in frequent_itemsets if not x[1] / total_num_transaction < support_count]
        all_frequent_itemsets.append(frequent_itemsets)
        k += 1

    return [x for x in all_frequent_itemsets if x]


def generate_candidates(itemsets, k):
    """takes a list of itemsets and generates itemsets of size k + 1"""
    new_itemsets = []

    # create a list of itemsets that are paired
    for i in range(len(itemsets) - 1):
        for j in range(i + 1, len(itemsets)):
            new_itemsets.append(itemsets[i][0].union(itemsets[j][0]))

    # keep itemsets that have size k + 1 and assign frequency value of 0
    frequent_itemsets = []
    for itemset in new_itemsets:
        if len(itemset) == k + 1 and not ([itemset, 0] in frequent_itemsets):
            frequent_itemsets.append([itemset, 0])

    return frequent_itemsets


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
print(supported_itemsets)



