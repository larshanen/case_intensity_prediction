import pandas as pd
from datetime import timedelta
from data_preprocessors.data_transformation import dates_converter

#  Define the function to match CBS data to feature dataset
def cbs_concatenator(original_df):
    # Load data into a df
    path = '../../Data/Regionale_kerncijfers_Nederland_09042024_095239.csv'
    cbs_df = pd.read_csv(path, delimiter=';', skiprows=4, header=0)

    # Remove the last row
    cbs_df = cbs_df.iloc[:-1]

    # Store new headers
    headers = ['gemeente',
            'jaar',
            'inwoners',
            'inwoners_per_km2',
            'bevolkingsgroei_per_1000',
            'uitkeringsontvangers']
    cbs_df.columns = headers

    # Replace commas with dots in the specified columns
    cbs_df['bevolkingsgroei_per_1000'] = cbs_df['bevolkingsgroei_per_1000'].str.replace(',', '.')

    # List all columns to be casted
    convert_dict_cbs = {'bevolkingsgroei_per_1000': 'float64',
                        'inwoners': 'float64'}

    # Cast columns
    cbs_df = cbs_df.astype(convert_dict_cbs)

    # Transform 'uiterkingsontvangers' to 'uitkeringsontvangers_per_1000'
    cbs_df['uitkeringsontvangers_per_1000'] = cbs_df['uitkeringsontvangers'] / (cbs_df['inwoners'] / 1000)

    # Delete 'uitkeringsontvangers' column
    cbs_df.drop(columns=['uitkeringsontvangers'], inplace=True)
    
    # Store columns to be copied
    cbs_df_columns = cbs_df.drop(columns=['jaar', 'gemeente'])

    # Add new columns to new_df if they don't exist
    for column in cbs_df_columns.columns:
        original_df[column] = 0.0
    
    for index, row in original_df.iterrows():
        gemeente_case = row['Gemeente']
        jaar_case = row['Datum_melding'].year

        if not pd.isna(gemeente_case) and gemeente_case != 'outside_WB': # All remains 0 for these, problematic?
            # Find corresponding row in cbs_df
            cbs_row = cbs_df[(cbs_df['gemeente'] == gemeente_case) & (cbs_df['jaar'] == jaar_case)]

            # Assign values from cbs_row to new_df
            for column in cbs_df_columns.columns:
                while not cbs_row.empty and pd.isna(cbs_row[column].iloc[0]) and jaar_case > 2018:
                    jaar_case -= 1
                    cbs_row = cbs_df[(cbs_df['gemeente'] == gemeente_case) & (cbs_df['jaar'] == jaar_case)]
                if not cbs_row.empty and not pd.isna(cbs_row[column].iloc[0]):
                    original_df.at[index, column] = cbs_row[column].iloc[0]
    return original_df

# Define the function to extract workload in previous 7 days
def workload_encoder(preprocessed_df, global_df):
    # Preprocess intervention dates in global_df
    global_df['Datum'] = global_df['Datum'].apply(dates_converter)

    # For every row in preprocessed_df, calculate the workload in the previous 7 days
    for index, row in preprocessed_df.iterrows():
        # Get notification date and interventionist
        notification_date = row['Datum_melding']
        interventionist = row['Medewerker']

        # Get dates of the prior 7 days
        last_week_dates = pd.date_range(end=notification_date - timedelta(days=1), periods=7, freq='D')

        # Get the number of interventions in the last week for the entire team
        frequencies_intervention_dates_team = global_df['Datum'].explode().value_counts().sort_index()
        frequencies_intervention_dates_team.index = pd.to_datetime(frequencies_intervention_dates_team.index)
        frequencies_intervention_dates_team = frequencies_intervention_dates_team.reindex(last_week_dates, fill_value=0)
        last_week_interventions_team = frequencies_intervention_dates_team.sum()
        preprocessed_df.at[index, 'workload_week_interventions_team'] = last_week_interventions_team

        # Get the number of cases in the last week for the entire team
        frequencies_case_dates_team = global_df['Datum melding'].value_counts().sort_index()
        frequencies_case_dates_team.index = pd.to_datetime(frequencies_case_dates_team.index)
        frequencies_case_dates_team = frequencies_case_dates_team.reindex(last_week_dates, fill_value=0)
        last_week_cases_team = frequencies_case_dates_team.sum()
        preprocessed_df.at[index, 'workload_week_cases_team'] = last_week_cases_team

        # Get the number of interventions in the last week for the corresponding interventionist
        frequencies_intervention_dates_interventionist = global_df[global_df['Medewerker'] == interventionist]['Datum'].explode().value_counts().sort_index()
        frequencies_intervention_dates_interventionist.index = pd.to_datetime(frequencies_intervention_dates_interventionist.index)
        frequencies_intervention_dates_interventionist = frequencies_intervention_dates_interventionist.reindex(last_week_dates, fill_value=0)
        last_week_interventions_interventionist = frequencies_intervention_dates_interventionist.sum()
        preprocessed_df.at[index, 'workload_week_interventions_interventionist'] = last_week_interventions_interventionist

        # Get the number of cases in the last week for the corresponding interventionist
        frequencies_case_dates_interventionist = global_df[global_df['Medewerker'] == interventionist]['Datum melding'].value_counts().sort_index()
        frequencies_case_dates_interventionist.index = pd.to_datetime(frequencies_case_dates_interventionist.index)
        frequencies_case_dates_interventionist = frequencies_case_dates_interventionist.reindex(last_week_dates, fill_value=0)
        last_week_cases_interventionist = frequencies_case_dates_interventionist.sum()
        preprocessed_df.at[index, 'workload_week_cases_interventionist'] = last_week_cases_interventionist

        # Preprocess dates in global_df
        global_df['Datum melding'] = pd.to_datetime(global_df['Datum melding'])
        global_df['Datum afsluiting'] = pd.to_datetime(global_df['Datum afsluiting'])

        # Get the number of ongoing cases at the date of notification for the entire team
        ongoing_cases_team = global_df[(global_df['Datum melding'] <= notification_date) & 
                                       (global_df['Datum afsluiting'].isnull() | (global_df['Datum afsluiting'] > notification_date))].shape[0]
        preprocessed_df.at[index, 'workload_ongoing_cases_team'] = ongoing_cases_team

        # Get the number of ongoing cases at the date of notification for the corresponding interventionist
        ongoing_cases_interventionist = global_df[(global_df['Datum melding'] <= notification_date) & 
                                                  (global_df['Medewerker'] == interventionist) & 
                                                  (global_df['Datum afsluiting'].isnull() | (global_df['Datum afsluiting'] > notification_date))].shape[0]
        preprocessed_df.at[index, 'workload_ongoing_cases_interventionist'] = ongoing_cases_interventionist

    return preprocessed_df