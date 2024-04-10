# IMPORT LIBRARIES
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import pyodbc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import median_absolute_error

# CONNECT TO SERVER TO RETRIEVE DF
# Replace server name if source changes
server = 'AZ-ODB0\ODBWB'
database = 'OnderzoeksDB_WB'
# Create a connection string
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database}'
# Establish connection
connection = pyodbc.connect(connection_string)
# Create SQL query string
query = 'SELECT * FROM Regas_Problematiek'
# Execute the query and fetch the data into a Pandas DataFrame
df_og = pd.read_sql(query, connection)
# Close the connection
connection.close()

# PREPROCESS DF
# Reorder columns and make copy of df
df = df_og[['Casus',
            'Persoonsnummer',
            'Gemeente',
            'JaarStartCasus',
            'Datum melding',
            'Geboortejaar',
            'Geboortemaand',
            'Soort melder',
            'Aanwezigheid minderjarige kinderen',
            'Is er sprake van huiselijk geweld?',
            'Is er sprake van agressie / geweld?',
            'Inhoud melding',
            'E33 melding',
            'Wijk-GGD',
            'Melding hoe ontvangen',
            'Vermoeden van problematiek',
            'Eerste advies',
            'Medewerker',   

            'Hoofdprobleem (1 antwoord mogelijk)',
            'Subproblemen (meerdere antwoorden mogelijk)',
            'Besproken op datum',

            'Inhoud',
            'Contact met',
            'Soort contact',
            'Datum',

            'Datum afsluiting',
            'Reden afsluiting',
            'Voortgang proces door',
            'Bijzonderheden',
            'Soort instelling verwijzing',

            'DubbelRecord']].copy()
# Drop all rows that have not been closed yet, without notification dates and with condition 'Onverzekerd'
df.dropna(subset=['Datum afsluiting'], inplace=True)
df.dropna(subset=['Datum melding'], inplace=True)
df.drop(df[df['Vermoeden van problematiek'] == 'Onverzekerde zorg GGD GHOR NL'].index, inplace=True)
# Define function to preprocess boolean-like values for casting
def convert_to_bool(column):
    bool_mapping = {'y': True, 'Ja': True, 'n': False, 'Nee': False}
    return bool_mapping.get(column, np.NaN)
# List all columns to be casted
columns_to_convert = ['Aanwezigheid minderjarige kinderen', 
                      'Is er sprake van huiselijk geweld?', 
                      'Is er sprake van agressie / geweld?', 
                      'E33 melding', 
                      'Wijk-GGD', 
                      'DubbelRecord']
# Cast columns
for col in columns_to_convert:
    df[col] = df[col].apply(convert_to_bool)
# List all columns to be casted
convert_dict = {'Casus': object,
                'Persoonsnummer': object,
                'Geboortejaar': object,
                'Geboortemaand': object,
                'JaarStartCasus': object,
                
                'Gemeente': 'category',
                'Soort melder': 'category',
                'Melding hoe ontvangen': 'category',
                'Eerste advies': 'category',
                'Medewerker': 'category',
                'Hoofdprobleem (1 antwoord mogelijk)': 'category',
                'Reden afsluiting': 'category',
                'Voortgang proces door': 'category',
                'Soort instelling verwijzing': 'category',

                'Aanwezigheid minderjarige kinderen': 'boolean',
                'Is er sprake van huiselijk geweld?': 'boolean',
                'Is er sprake van agressie / geweld?': 'boolean',
                'E33 melding': 'boolean',
                'Wijk-GGD': 'boolean',
                'DubbelRecord': 'boolean'
                }
# Cast columns
df = df.astype(convert_dict)
# Change date column Dtypes
df[['Datum afsluiting',
    'Datum melding',
    'Besproken op datum']] = df[['Datum afsluiting',
                                 'Datum melding',
                                 'Besproken op datum']].apply(pd.to_datetime)
# Rename date columns for .apply function on multiple column names
df.rename(columns={"Datum afsluiting": "Datum_afsluiting",
                   "Datum melding": "Datum_melding",
                   "Besproken op datum": "Besproken_op_datum"}, inplace=True)
# Function to convert string of dates to list of datetime objects
def string_to_dates(date_string):
    # Return empty list if no intervention dates are registered
    if not date_string:
        return []
    # Split by comma's and add to list
    else:
        date_list = date_string.split(', ')
        date_list = [datetime.strptime(date, '%Y-%m-%d').date() for date in date_list]
        return date_list
# Overwrite the column with the transformed version
df['Datum'] = df['Datum'].apply(string_to_dates)
# Function to convert string to list of recurring intervention columns
def string_to_list(string):
    # Return empty list if null
    if not string:
        return []
    # Split by comma's and add to list
    else:
        list = string.split(', ')
        return list
# Overwrite the column with the transformed version
df['Soort contact'] = df['Soort contact'].apply(string_to_list)
df['Contact met'] = df['Contact met'].apply(string_to_list)
# Create X and y where feature engineered columns can be stored
X = pd.DataFrame()
y = pd.DataFrame()
# Function to convert string of dates to list of datetime objects
def get_n_businessdays(date_notification, dates_interventions):
    if not dates_interventions:
        return 0 #This actually has to be None, but RF regressor doesn't support this, be aware of that!
    else:
        werkdagen = pd.bdate_range(date_notification, max(dates_interventions)).shape[0]
        return int(werkdagen)
# Add the calculated days to a new column
y['Dagen_tot_laatst'] = df.apply(lambda x: get_n_businessdays(x.Datum_melding, x.Datum), axis=1)
# Define the function to calculate the amount of interventions
def get_n_interventions(dates):
    if not dates:
        return 0
    else:
        return len(dates)
# Add the calculated amount of interventions to a new column
y['No_interventions'] = df['Datum'].apply(get_n_interventions)
# Define the function to calculate the age at notification date
def get_age(birth_year, birth_month, notification_date):
    notification_year = notification_date.year
    notification_month = notification_date.month
    age = notification_year - birth_year
    if notification_month < birth_month:
        age -= 1
    return age
# Add the calculated ages to a new column
X['Leeftijd_melding'] = df.apply(lambda x: get_age(x.Geboortejaar, x.Geboortemaand, x.Datum_melding), axis=1)
# Define function to reduce skewness, assuming positive skewness starting the downsampling from class '0'
def reduce_skew(X, y, imbalanced_column: str, final_heavy_class: int, downsampling_perc_heaviest):
    # Create series to store all frequencies
    classes = pd.Series(range(int(y[imbalanced_column].max() + 1)))
    counts = y[imbalanced_column].value_counts().sort_index()
    frequencies = counts.reindex(classes, fill_value=0)

    # Disregard classes exceeding final_heavy_class
    classes_to_downsample = frequencies[:final_heavy_class+1]

    # Calculate median class size for disregarded classes
    subseq_classes = frequencies[final_heavy_class+1:final_heavy_class+round(final_heavy_class*0.25)]
    median_subseq_class_size = np.median(subseq_classes.tolist())

    # Define downsampling percentages for all downsampling classes
    downsampling_percs = []
    indices_removed = []
    max_class_downsample = max(classes_to_downsample) - median_subseq_class_size
    for class_label, class_size in classes_to_downsample.items():
        if class_size > median_subseq_class_size:
            class_size = class_size - median_subseq_class_size
        downsampling_perc = (class_size * downsampling_perc_heaviest) / max_class_downsample
        downsampling_percs.append(downsampling_perc)

        # Calculate amount of instances to be removed
        n = round(class_size * downsampling_perc)
        
        # Randomly choose instances to remove (should this have a random state?)
        class_indices_candidates = y[y[imbalanced_column] == class_label].index
        class_indices_remove = np.random.choice(class_indices_candidates, size=int(n), replace=False)
        indices_removed.extend(class_indices_remove)

    # Remove all indices in indices_to_remove from df with inplace=True argument
    y.drop(indices_removed, inplace=True)
    X.drop(indices_removed, inplace=True)
    return X, y
X_reduced, y_reduced = reduce_skew(X, y, 'Dagen_tot_laatst', 25, 0.5)
# Remove the last 15 rows from both X and y
X_train = X.iloc[:-15]
y_train = y.iloc[:-15]
# Store the removed rows in new DataFrames
X_test = X.iloc[-15:]
y_test = y.iloc[-15:]
# Fit RF regressor
RF = RandomForestRegressor(random_state=42)
RF.fit(X_train, y_train)
# Predict test set
y_pred = RF.predict(X_test)
# Store MAD's
y_pred_df = pd.DataFrame(y_pred,
                         columns=['Dagen_tot_laatst', 'No_interventions'])

# Calculate median absolute deviation (MAD) for each output separately
mad_interventions = median_absolute_error(y_test['No_interventions'], y_pred_df['No_interventions'])
mad_days = median_absolute_error(y_test['Dagen_tot_laatst'], y_pred_df['Dagen_tot_laatst'])

# WEB APP UI
st.title('OGGZ Case intensiteit voorspelling')
st.subheader('Voor binnenkomende cases')

# Ask visitor for preferred outcome presentation
preferred_outcome = st.selectbox('Ik wil de case-intensiteit zien van:',
                                 ('Kies een optie...',
                                  'Een open case naar keuze',
                                  'Een specifiek teamlid',
                                  'De totale caseload van het team'))

# Get predictions of one of predefined cases
case_names = ['Emma Haldane',
              'Lucas Thorne',
              'Isabella Fontaine',
              'Oliver Sinclair',
              'Sophia Drake',
              'Liam Harrington',
              'Ava Kingsley',
              'Noah Hawthorne',
              'Mia Winslow',
              'Ethan Blackwood',
              'Amelia Sterling',
              'Jacob Whitaker',
              'Charlotte Mercer',
              'William Kensington',
              'Harper Donovan']

if preferred_outcome == 'Een open case naar keuze':
    case = st.selectbox(
        'Kies één van de cases',
        case_names)
    interventies = round(y_pred[case_names.index(case)][1])
    dagen = round(y_pred[case_names.index(case)][0])
    einddatum = date.today() + timedelta(days=dagen)

    # Prepare data to show case information table
    st.subheader(f'Case informatie **{case}**:')
    data = {
        'Kenmerk': ['Gemeente', 'Geboortejaar', 'Geboortemaand', 'Datum melding', 'Soort melder', 'Eerste advies', 'Aanwezigheid minderjarige kinderen', 'Is er sprake van huiselijk geweld?', 'Is er sprake van agressie / geweld?', 'Vermoeden van problematiek'],
        'Waarde': [
            df_og['Gemeente'][X_test.index[case_names.index(case)]],
            df_og['Geboortejaar'][X_test.index[case_names.index(case)]],
            df_og['Geboortemaand'][X_test.index[case_names.index(case)]],
            df_og['Datum melding'][X_test.index[case_names.index(case)]],
            df_og['Soort melder'][X_test.index[case_names.index(case)]],
            df_og['Eerste advies'][X_test.index[case_names.index(case)]],
            df_og['Aanwezigheid minderjarige kinderen'][X_test.index[case_names.index(case)]],
            df_og['Is er sprake van huiselijk geweld?'][X_test.index[case_names.index(case)]],
            df_og['Is er sprake van agressie / geweld?'][X_test.index[case_names.index(case)]],
            df_og['Vermoeden van problematiek'][X_test.index[case_names.index(case)]]]}
    df = pd.DataFrame(data)
    st.table(df)

    st.subheader(f'**Voorspelling**:')
    st.write(f'Deze case behoeft naar voorspelling **{interventies} ± {round(mad_interventions)}** interventies verspreid over **{dagen} ± {round(mad_days)} werkdagen**.')

# Get predictions of caseload for an interventionist
interventionist_names = ['Sanne',
                         'Timo',
                         'Annelies',
                         'Pieter',
                         'Lotte',
                         'Jan',
                         'Adriaan']

if preferred_outcome == 'Een specifiek teamlid':
    interventionist = st.selectbox(
        'Kies één medewerkers',
        interventionist_names)
    # Add when X contains interventionist column
    st.subheader(f'Voorspelling caseload **{interventionist}**:')

# Get predictions of caseload of team as a whole
if preferred_outcome == 'De totale caseload van het team':
    caseload_interventies = 0
    for case in range(len(y_pred)):
        caseload_interventies += y_pred[case][1]

    longest_case = round(np.max(y_pred[: -1])) #indexing not right yet, doesn't seem to correctly extract longest case
    longest_enddate = date.today() + timedelta(days=longest_case)

    st.subheader('Caseload team:')
    st.write(f'Deze cases leiden tot een caseload van **{round(caseload_interventies)}** interventies tussen nu en **{longest_enddate}**.')