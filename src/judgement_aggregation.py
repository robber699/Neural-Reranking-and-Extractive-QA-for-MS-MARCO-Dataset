## implement part 1 here
import sys
import pandas as pd
import os


# group - a group of judgements that belong to a document, query pair
def judgement_aggregation(group):
    # print(group[['documentId', 'queryId', 'relevanceLevel', 'userId', 'weight']])

    # calculate the average time spent on judging the doc, query pair
    average_judgement_time = group['durationUsedToJudgeMs'].mean()

    # threshold for average time: if < avg - threshold, then reduce weight of annotator
    threshold = 0.3

    # Iterate over the group and update the weights
    for index, row in group.iterrows():
        if row['durationUsedToJudgeMs'] < average_judgement_time - (average_judgement_time * threshold):
            # Reduce the weight of the annotator
            # print("Reducing weight for user: " + row['userId'])
            group.at[index, 'weight'] *= 0.8

    # Group by 'relevanceLevel' and sum 'weight' column
    grouped = group.groupby('relevanceLevel')['weight'].sum()

    # Get the 'relevanceLevel' with the maximum weight
    max_weight_relevance = grouped.idxmax()

    # return the graded relevance
    if max_weight_relevance == '0_NOT_RELEVANT':
        return 0
    elif max_weight_relevance == '1_TOPIC_RELEVANT_DOES_NOT_ANSWER':
        return 1
    elif max_weight_relevance == '2_GOOD_ANSWER':
        return 2
    elif max_weight_relevance == '3_PERFECT_ANSWER':
        return 3
    else:
        return -1


def create_i_freq_table(raw_judgements) -> pd.DataFrame:
    # create a table with the number of occurrences of each relevance level for each user
    # table columns: userId, |0_NOT_RELEVANT| ...
    rel_freq: pd.DataFrame = pd.crosstab(raw_judgements['userId'], raw_judgements['relevanceLevel'])

    # Reset the index to get 'userId' as a column
    rel_freq.reset_index(inplace=True)
    # rename the columns (to get rid of the 'relevanceLevel' prefix)
    rel_freq.columns = ['userId', '0_NOT_RELEVANT', '1_TOPIC_RELEVANT_DOES_NOT_ANSWER', '2_GOOD_ANSWER',
                        '3_PERFECT_ANSWER']

    # Calculate the total number of annotations for each user
    rel_freq['total_judgements'] = rel_freq.drop(columns=['userId']).sum(axis=1)

    # Calculate frequency of each relevance level for each user
    rel_freq['freq_0'] = rel_freq['0_NOT_RELEVANT'] / rel_freq['total_judgements']
    rel_freq['freq_1'] = rel_freq['1_TOPIC_RELEVANT_DOES_NOT_ANSWER'] / rel_freq['total_judgements']
    rel_freq['freq_2'] = rel_freq['2_GOOD_ANSWER'] / rel_freq['total_judgements']
    rel_freq['freq_3'] = rel_freq['3_PERFECT_ANSWER'] / rel_freq['total_judgements']

    # print(rel_freq)
    return rel_freq


def create_global_freq_table(rel_freq: pd.DataFrame) -> pd.DataFrame:
    columns_to_sum = ['0_NOT_RELEVANT', '1_TOPIC_RELEVANT_DOES_NOT_ANSWER', '2_GOOD_ANSWER', '3_PERFECT_ANSWER',
                      'total_judgements']

    # Sum the selected columns
    sums = rel_freq[columns_to_sum].sum()
    # print(sums)

    # Calculate global frequencies
    global_freq = pd.DataFrame({
        'global_sum': sums,
        'global_freq': sums / sums['total_judgements']
    })

    # Reset the index to get 'relevanceLevel' as a column
    global_freq.reset_index(inplace=True)

    # Rename the index column to 'relevanceLevel'
    global_freq.rename(columns={'index': 'relevanceLevel'}, inplace=True)

    # we don't need the 'total_judgements' row anymore
    # Create a boolean mask where True indicates the rows we want to keep
    mask = global_freq['relevanceLevel'] != 'total_judgements'

    # Apply the mask to the DataFrame
    global_freq = global_freq[mask]

    # print(global_freq)
    return global_freq


def aggregation_process(path):
    print("Loading raw judgements...")
    raw_judgements = pd.read_csv(path, sep='\t')

    # create individual frequency table
    user_freq = create_i_freq_table(raw_judgements)

    # create global frequency table
    global_freq = create_global_freq_table(user_freq)

    # Merging frequency tables with judgement table to facilitate computation of weights
    # Merge raw_judgements with user_freq on 'userId'
    raw_judgements = pd.merge(raw_judgements, user_freq, on='userId', how='left')

    # Merge raw_judgements with global_freq on 'relevanceLevel'
    raw_judgements = pd.merge(raw_judgements, global_freq, on='relevanceLevel', how='left')

    print("Start computing judgement weights...")

    # add a new column "weight"
    # based on user_freq, global_freq, calculate the weight
    # formular used: difference-based BCR wik = 1 + freq(k) - freqi(k)
    for i in range(4):
        raw_judgements.loc[raw_judgements['relevanceLevel'].str.startswith(f'{i}_'), 'weight'] = 1 + raw_judgements[
            'global_freq'] - raw_judgements[f'freq_{i}']

    print("Start aggregating judgements...")
    aggregated_judgements = raw_judgements.groupby(['queryId', 'documentId']).apply(judgement_aggregation).reset_index()

    print("Preparing output of aggregated judgements...")

    # Add the new column to the DataFrame
    aggregated_judgements['hardcoded-Q0'] = "Q0"  # Replace 'some_values' with the actual values you want to add

    # Reorder the columns
    aggregated_judgements = aggregated_judgements[['queryId', 'hardcoded-Q0', 'documentId', 0]]

    # Rename the column for relevance grade
    aggregated_judgements.rename(columns={0: 'relevance-grade'}, inplace=True)

    # print(raw_judgements)
    # print(aggregated_judgements)

    print("Saving the aggregated judgements to a TSV file...")
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    aggregated_judgements.to_csv(output_dir + '/new_aggregated_judgements.tsv', sep='\t', index=False)


if __name__ == '__main__':
    # py judgement_aggregation.py ../../air-exercise-2/Part-1/fira-22.judgements-anonymized.tsv
    if len(sys.argv) == 2:
        aggregation_process(sys.argv[1])
    else:
        print("Usage: <raw_judgements>")
