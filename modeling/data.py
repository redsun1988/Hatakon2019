import pandas as pd
import re


def remove_tags(text):
    return re.sub('<[^<]+?>', ' ', text)


def remove_multiple_spaces(text):
    return re.sub(' +', ' ', text)


def clean_column(df, col_name):
    df[col_name] = df[col_name].str.replace('<br>', ' ')
    df[col_name] = df[col_name].str.replace('<br />', ' ')
    df[col_name] = df[col_name].apply(remove_tags)
    df[col_name] = df[col_name].apply(remove_multiple_spaces)
    df[col_name] = df[col_name].str.replace('&nbsp;', ' ')
    df[col_name] = df[col_name].str.replace('\n', ' ')
    df[col_name] = df[col_name].str.replace('\t', ' ')
    df[col_name] = df[col_name].str.strip()


def read_csv(file_path, text_col_name, head_col_name, rows_fraction=100):
    DataSet = pd.read_csv(file_path, encoding="ansi")

    DataSet = DataSet[[text_col_name, head_col_name]]

    DataSet.dropna(inplace=True)

    clean_column(DataSet, text_col_name)
    clean_column(DataSet, head_col_name)

    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    DataSet[text_col_name] = '\t' + DataSet[text_col_name] + '\n'

    if (rows_fraction != 100):
        DataSet = DataSet.head(int(len(DataSet)*(rows_fraction/100)))

    max_input_len = DataSet[text_col_name].str.len().max()
    max_output_len = DataSet[head_col_name].str.len().max()

    texts = DataSet[text_col_name].values
    headlines = DataSet[head_col_name].values

    print('Examples number:', len(DataSet))

    del DataSet

    return texts, headlines, max_input_len, max_output_len
