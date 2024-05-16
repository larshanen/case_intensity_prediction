import pandas as pd
from datetime import datetime
from tabulate import tabulate

def data_splitter(df, targets):
    # Define cutoff values for train-validation-test splits based on year
    train_cutoff = 2020
    val_cutoff = 2021
    test_cutoff = 2022

    # Drop records with missing values in 'Datum_afsluiting' column
    df.dropna(subset=['Datum_afsluiting'], inplace=True)

    # Extract years of notifications
    df['year_Datum_melding'] = pd.to_datetime(df['Datum_melding']).dt.year

    # Filter indices based on the cutoff values
    train_indices = df.loc[df['year_Datum_melding'] <= train_cutoff].index
    val_indices = df.loc[(df['year_Datum_melding'] > train_cutoff) & (df['year_Datum_melding'] <= val_cutoff)].index
    test_indices = df.loc[(df['year_Datum_melding'] > val_cutoff) & (df['year_Datum_melding'] <= test_cutoff)].index
    oos_indices = df.loc[df['year_Datum_melding'] > test_cutoff].index

    # Drop df['year_Datum_melding'] column
    df.drop(columns=['year_Datum_melding'], inplace=True)

    # Replace missing values with today's date
    #df['Datum_afsluiting'].fillna(datetime.today(), inplace=True) # RETHINK WHEN CONSTRUCTING EVALUATION METHOD

    # Split into train, validation, and test sets
    train, val, test, oos = df.loc[train_indices], df.loc[val_indices], df.loc[test_indices], df.loc[oos_indices]

    # Split data into features and targets
    X_train, X_val, X_test, X_oos = train.drop(targets, axis=1), val.drop(targets, axis=1), test.drop(targets, axis=1), oos.drop(targets, axis=1)
    y_train, y_val, y_test, y_oos = train[targets], val[targets], test[targets], oos[targets]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_oos, y_oos

def data_splitter_v2(df, targets):
    # Define cutoff values for train-validation-test splits based on year
    train_cutoff = 2021
    val_cutoff = 2022

    # Extract years of notifications
    df['year_Datum_melding'] = pd.to_datetime(df['Datum_melding']).dt.year

    # Filter indices based on the cutoff values
    train_indices = df.loc[df['year_Datum_melding'] <= train_cutoff].index
    val_indices = df.loc[(df['year_Datum_melding'] > train_cutoff) & (df['year_Datum_melding'] <= val_cutoff)].index
    
    # Get unique 'Medewerker' values in both training and validation sets
    unique_medewerkers_train = set(df.loc[train_indices, 'Medewerker'].unique())
    unique_medewerkers_val = set(df.loc[val_indices, 'Medewerker'].unique())

    # Filter test set indices where 'Medewerker' values are unseen in both training and validation sets
    test_indices = df.loc[(df['year_Datum_melding'] > val_cutoff) & (df['Medewerker'].isin(unique_medewerkers_train | unique_medewerkers_val))].index
    oos_indices = df.loc[(df['year_Datum_melding'] > val_cutoff) & (~df['Medewerker'].isin(unique_medewerkers_train | unique_medewerkers_val))].index

    # Drop df['year_Datum_melding'] column
    df.drop(columns=['year_Datum_melding'], inplace=True)

    # Split into train, validation, and test sets
    train, val, test, oos = df.loc[train_indices], df.loc[val_indices], df.loc[test_indices], df.loc[oos_indices]

    
    # Split data into features and targets
    X_train, X_val, X_test, X_oos = train.drop(targets, axis=1), val.drop(targets, axis=1), test.drop(targets, axis=1), oos.drop(targets, axis=1)
    y_train, y_val, y_test, y_oos = train[targets], val[targets], test[targets], oos[targets]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_oos, y_oos

def print_characteristics(train, dev, test, oos):
    # Calculate the number of months in each split
    num_months_train = len(train['Datum_melding'].dt.to_period('Y').unique()) * 12
    num_months_dev = len(dev['Datum_melding'].dt.to_period('Y').unique()) * 12
    num_months_test = len(test['Datum_melding'].dt.to_period('Y').unique()) * 12
    num_months_oos = len(oos['Datum_melding'].dt.to_period('Y').unique()) * 12

    # Identify 'Medewerker' instances in each split
    train_interventionists = set(train['Medewerker'].unique())
    dev_interventionists = set(dev['Medewerker'].unique())
    test_interventionists = set(test['Medewerker'].unique())
    oos_interventionists = set(oos['Medewerker'].unique())

    # Identify unseen 'Medewerker' instances in each split
    train_unseen_interventionists = train_interventionists
    dev_unseen_interventionists = dev_interventionists - train_interventionists
    test_unseen_interventionists = test_interventionists - (train_interventionists | dev_interventionists)
    oos_unseen_interventionists = oos_interventionists - (train_interventionists | dev_interventionists | test_interventionists)

    # Calculate the percentage of instances with unseen 'Medewerker' for each split
    train_percentage = (train['Medewerker'].isin(train_unseen_interventionists).sum() / len(train)) * 100
    dev_percentage = (dev['Medewerker'].isin(dev_unseen_interventionists).sum() / len(dev)) * 100
    test_percentage = (test['Medewerker'].isin(test_unseen_interventionists).sum() / len(test)) * 100
    oos_percentage = (oos['Medewerker'].isin(oos_unseen_interventionists).sum() / len(oos)) * 100

    # Create dictionaries for results
    amount_instances = {'train': len(train), 'dev': len(dev), 'test': len(test), 'OoS': len(oos)}
    amount_unseen_interventionists = {'train': len(train_unseen_interventionists), 'dev': len(dev_unseen_interventionists), 'test': len(test_unseen_interventionists), 'OoS': len(oos_unseen_interventionists)}
    percentages = {'train': train_percentage, 'dev': dev_percentage, 'test': test_percentage, 'OoS': oos_percentage}
    avg_instances_per_month = {'train': amount_instances['train'] / num_months_train / len(train_interventionists), 'dev': amount_instances['dev'] / num_months_dev / len(dev_interventionists), 'test': amount_instances['test'] / num_months_test / len(test_interventionists), 'OoS': amount_instances['OoS'] / num_months_oos / len(oos_interventionists)}

    # Store the amount of cases where 'Datum_afsluiting' is missing
    unclosed_cases = {'train': train['Datum_afsluiting'].isna().sum(), 'dev': dev['Datum_afsluiting'].isna().sum(), 'test': test['Datum_afsluiting'].isna().sum(), 'OoS': oos['Datum_afsluiting'].isna().sum()}

    # Create a list of tuples for tabular data
    table_data = [
        ("Set", "Year(s)", "# Cases", "# Unclosed cases", "# Unseen ints", "% Unseen ints' cases", "Incoming cases per month per int"),
        ("Train", "2019-2021", amount_instances["train"], unclosed_cases["train"], amount_unseen_interventionists["train"], f"{percentages['train']:.2f}%", avg_instances_per_month["train"]),
        ("Dev", "2022", amount_instances["dev"], unclosed_cases["dev"], amount_unseen_interventionists["dev"], f"{percentages['dev']:.2f}%", avg_instances_per_month["dev"]),
        ("Test", "2023-2024", amount_instances["test"], unclosed_cases["test"], amount_unseen_interventionists["test"], f"{percentages['test']:.2f}%", avg_instances_per_month["test"]),
        ("OoS", "2023-2024", amount_instances["OoS"], unclosed_cases["OoS"], amount_unseen_interventionists["OoS"], f"{percentages['OoS']:.2f}%", avg_instances_per_month["OoS"])
    ]

    # Return the results in tabular format
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
