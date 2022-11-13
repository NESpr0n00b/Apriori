#####
#   Mikhailangelo Panzo
#   Comsci 35 - Data Mining
#
#   This python code is a Association Rule Mining Algorithm based on Apriori
#
#   format of dataset:
#   csv file
#   header of item(s), item 1, item 2, item 3, ..., item n
#   n = the number of items of the itemset with the most items in the dataset
#   first column represents the frequency of the set of item in the row
#   rest of columns represent an item, filling the second column first then the third and so on
#   if there are no more items to be filled in the remaining column, leave them empty or null
#
#   example dataset:
#   Item(s),Item 1,Item 2,Item 3,Item 4
#   4,citrus fruit,semi-finished bread,margarine,ready soups
#   3,tropical fruit,yogurt,coffee,,
#   1,whole milk,,,,
#
#   steps present in this code:
#   1. extract data from csv to 2d array
#   2. set support and confidence
#   3. categorize 2d array to numerical indicators
#   4. create an array of dictionary containing itemset-frequency
#   5. iterate counting, purging, and combining by:
#       a. listing all possible n-itemset based on previous itemsets,
#           if there are none, end iteration
#       b. counting the frequency of itemset and assigning it to the corresponding key
#       c. deleting the itemset if it does not fall under support condition
#       d. raising n by 1, where n is 1 in first iteration
#   6. extract discovery sets to be used, or end entire code if no rules can be made
#   7. generate possible rules
#   8. keep rules that fit confidence
#   9. export rules
#####

import csv
import itertools
import os.path
import sys


def extract_data(inputted_filename):
    """Extracts data from a csv file named [inputted_filename] into a dictionary of data and header"""
    data = []  # main data to work with

    if not os.path.exists(inputted_filename + ".csv"):
        print("File does not exist.")
        sys.exit(0)

    with open(inputted_filename + ".csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            data.append(row)

    file.close()
    return {'data': data, 'header': header}


def set_amount(message):
    """Sets amount from 0 to 100 percent and prompts using [message] string"""

    choice = input(message + ". Enter percentage (0 - 100): ")
    while True:
        is_valid = True

        try:
            choice = float(choice)
        except:
            is_valid = False

        if is_valid:
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


def count_transactions(transaction_dataset):
    """counts the number of transactions from [transaction_data] and returns the total count"""

    total_count = 0
    for transaction in transaction_dataset:
        total_count += transaction[0]

    return total_count


def apriori_support(transaction_data, min_support, total_num_transaction):
    """purges possible itemsets based on support count and returns an array of supported itemsets with rows of
    supported n-itemsets """

    min_support = min_support / 100

    dataset = transaction_data["data"].copy()
    all_frequent_itemsets = []

    # first layer itemset-frequency , frequent_itemsets have itemset and frequency in list
    itemsets = []
    for category_value in range(len(transaction_data["reference"])):
        itemsets.append([{category_value}, 0])
    for transaction in dataset:
        for item_index in range(1, len(transaction)):
            itemsets[transaction[item_index]][1] += transaction[0]
    frequent_itemsets = [x for x in itemsets if min_support <= x[1] / total_num_transaction]
    all_frequent_itemsets.append(frequent_itemsets)
    # print(frequent_itemsets)

    # apriori algorithm
    k = 1
    while True:
        indices_to_be_deleted = []
        # candidate itemsets do not have frequencies
        frequent_itemsets = generate_candidates(frequent_itemsets, k)
        # breaks loop if no more itemsets can be made
        if len(frequent_itemsets) == 0:
            break
        # for each transaction count them in frequent itemsets
        for transaction_index in range(len(dataset)):
            transaction_items = dataset[transaction_index].copy()
            transaction_items.pop(0)  # removes the column that specifies count of that itemset
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
        frequent_itemsets = [x for x in frequent_itemsets if min_support <= x[1] / total_num_transaction]
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


def extract_discovery_sets(all_frequent_itemsets):
    """takes frequency itemsets and extracts all itemsets that can form rules and returns it; returns empty if no pairs
    can be made """

    if len(all_frequent_itemsets) <= 1:
        return []

    top_k_sets = []
    lower_k_sets = []

    for itemset in all_frequent_itemsets[-1]:
        top_k_sets.append(itemset[0])

    if not len(all_frequent_itemsets) == 2:
        for itemset in all_frequent_itemsets[-2]:
            for top_k_set in top_k_sets:
                if not itemset[0].issubset(top_k_set):
                    lower_k_sets.append(itemset[0])

    return top_k_sets + lower_k_sets


def extract_rules(all_discovery_sets):
    """generates rules from [all_discovery_sets] and returns the rules in a list of 2 sets:
    set1 => set2 is represented by [{set1},{set2}]"""

    all_possible_rules = []
    for discovery_set in all_discovery_sets:
        set_list = list(discovery_set)

        antecedent_lists = []
        for antecedent_num in range(1, len(set_list) + 1):
            for combination in itertools.combinations(set_list, antecedent_num):
                antecedent_lists.append(combination)

        for antecedent_list in antecedent_lists:
            possible_consequent_list = set_list.copy()
            for item in antecedent_list:
                possible_consequent_list.remove(item)

            consequent_lists = []
            for consequent_num in range(1, len(possible_consequent_list) + 1):
                for combination in itertools.combinations(possible_consequent_list, consequent_num):
                    consequent_lists.append(combination)

            for consequent_list in consequent_lists:
                all_possible_rules.append([set(antecedent_list), set(consequent_list)])

    return all_possible_rules


def extract_confident_rules(all_possible_rules, itemset_frequencies, min_confidence, total_num_trans):
    """generates rules from [all_possible_rules] that are within confidence threshold
     and returns the rules in a list of 2 sets: set1 => set2 is represented by [{set1},{set2}]"""

    min_confidence = min_confidence / 100
    confident_rules = []

    for rule in all_possible_rules:
        antecedent = rule[0]
        union = antecedent.union(rule[1])

        antecedent_frequency = 0
        for itemset in itemset_frequencies[len(antecedent) - 1]:
            if itemset[0] == antecedent:
                antecedent_frequency = itemset[1]
                break

        union_frequency = 0
        for itemset in itemset_frequencies[len(union) - 1]:
            if itemset[0] == union:
                union_frequency = itemset[1]
                break

        if antecedent_frequency + union_frequency == 0:
            print("ERROR: faulty logic in itemsets in rules")
        else:
            rule_support = union_frequency / total_num_trans
            rule_confidence = union_frequency / antecedent_frequency
            if min_confidence <= rule_confidence:
                confident_rules.append([rule, rule_support, rule_confidence])

    return confident_rules


def export_rules(confident_rules, category_reference, inputted_filename, total_transactions, min_confidence,
                 min_support):
    """exports the rules to a text file named "[filename] association rules.txt"
    with details of the dataset, support, and confidence"""

    export_filename = inputted_filename + ' association rules.txt'
    if os.path.exists(export_filename):
        count = 1
        while True:
            export_filename = inputted_filename + ' association rules (' + str(count) + ').txt'
            if not os.path.exists(export_filename):
                break
            count += 1

    with open(export_filename, 'w') as file:
        file.write('Association Rules from the dataset ' + inputted_filename + '.csv\n')
        file.write(f'Total number of transactions: {total_transactions}\n')
        file.write(f'Minimum support threshold: {min_support}%\n')
        file.write(f'Minimum confidence threshold: {min_confidence}%\n\n')
        file.write('-----\n\n')
        file.write('Rules:\n')

        count = 0
        for rule_detail in confident_rules:
            rule = rule_detail[0]
            text_rule = "{"
            antecedent_items = list(rule[0])
            consequent_items = list(rule[1])
            for i in range(len(antecedent_items) - 1):
                text_rule += category_reference[antecedent_items[i]] + ", "
            text_rule += category_reference[antecedent_items[-1]] + "} --> {"
            for i in range(len(consequent_items) - 1):
                text_rule += category_reference[consequent_items[i]] + ", "
            text_rule += category_reference[consequent_items[-1]] + "}"
            file.write(f'{text_rule} ({round(rule_detail[1] * 100, 2)}%, {round(rule_detail[2] * 100, 2)}%)\n')
            count += 1

        file.write('\nTotal number of rules: ' + str(count))

    file.close()
    print("Rules exported successfully in '" + export_filename + "' in the same directory as this program")


# Main Code

# step 1: extract data from csv to 2d array
print("NOTE: csv file must be in the same directory as this program")
filename = input("Enter the file name of the csv file to be data mined for association rules: ")
transaction_table = extract_data(filename)

# step 2: prompt user to set support and confidence
support = set_amount("Enter the minimum support threshold")
confidence = set_amount("Enter the minimum confidence threshold")

# step 3: translate categories to numerical representation
processed_transaction_table = category_numerification(transaction_table['data'])
total_num_of_transaction = count_transactions(processed_transaction_table['data'])

# step 4 and 5: create an array of itemset-frequency
# results will be a 2d list, columns consist of all different support itemsets, rows categorizes them by k in k-itemset
# each support itemsets are in a form of a list of itemset (set of integers) and frequency (integer):
# [{itemset}, frequency]
supported_itemsets = apriori_support(processed_transaction_table, support, total_num_of_transaction)

# step 6: extract last two k-itemsets
discovery_sets = extract_discovery_sets(supported_itemsets)

if not discovery_sets:
    # if there are no discovery sets, no rules are made
    print("There are no rules with the given minimum support threshold.")
else:
    # step 7: generate possible rules
    possible_rules = extract_rules(discovery_sets)

    # step 8: keep rules under confidence
    rules = extract_confident_rules(possible_rules, supported_itemsets, confidence, total_num_of_transaction)

    # step 9: export rules
    if rules:
        export_rules(rules, processed_transaction_table['reference'], filename, total_num_of_transaction, confidence,
                     support)
    else:
        print("There are no rules with the given minimum confidence threshold.")