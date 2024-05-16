import numpy as np
import pandas as pd

def column_selector(df_og):
    # Reorder columns and make copy of df
    df = df_og[[#'Casus',
                #'Persoonsnummer',
                'Gemeente',                                         # OH, freq and/or geo encoding                                      #done
                #'JaarStartCasus',
                'Datum melding',                                    # Year as is, cyclical datetime encoding for months and days        #done, day of month AND week encoding
                'Geboortejaar',                                     # Use to construct age                                              #done
                'Geboortemaand',                                    # Use to construct age                                              #done
                'Soort melder',                                     # OH and/or freq encoding                                           #done
                'Aanwezigheid minderjarige kinderen',               # Dummy                                                             #done
                'Is er sprake van huiselijk geweld?',               # Dummy                                                             #done
                'Is er sprake van agressie / geweld?',              # Dummy                                                             #done
                #'Inhoud melding',
                #'E33 melding',
                'Wijk-GGD',                                         # Dummy                                                             #done
                'Melding hoe ontvangen',                            # OH and/or freq encoding                                           #done
                'Vermoeden van problematiek',                       # Multi OH and/or multi (?) freq encoding                           #done
                'Eerste advies',                                    # OH and/or freq encoding                                           #done

                # EINDE MELDINGSFASE

                #'Hoofdprobleem (1 antwoord mogelijk)',
                #'Subproblemen (meerdere antwoorden mogelijk)',
                #'Besproken op datum',
                #'Inhoud',
                #'Contact met',
                #'Soort contact',
                'Datum',
                'Medewerker',                                       # OH and/or freq encoding                                           #done

                # EINDE TRAJECT/ONDERZOEKSFASE

                'Datum afsluiting',
                #'Reden afsluiting',
                #'Voortgang proces door',
                #'Bijzonderheden',
                #'Soort instelling verwijzing',
                
                # EINDE SLUITINGSFASE

                'DubbelRecord'                                      # Dummy                                                             #done
                ]].copy() # All hashed columns are not needed
    return df

def boolean_mapper(df):
    # Define function to preprocess boolean-like values for casting
    def convert_to_bool(column):
        bool_mapping = {'y': True, 'Ja': True, 'n': False, 'Nee': False}
        return bool_mapping.get(column, np.NaN)

    # List all columns to be casted
    columns_to_convert = ['Aanwezigheid minderjarige kinderen', 
                        'Is er sprake van huiselijk geweld?', 
                        'Is er sprake van agressie / geweld?', 
                        #'E33 melding', 
                        'Wijk-GGD', 
                        'DubbelRecord']

    # Cast columns
    for col in columns_to_convert:
        df[col] = df[col].apply(convert_to_bool)
    return df

def dtype_caster(df):
    # List all columns to be casted
    convert_dict = {#'Casus': object,
                    #'Persoonsnummer': object,
                    'Geboortejaar': object,
                    'Geboortemaand': object,
                    #'JaarStartCasus': object,
                    
                    'Gemeente': 'category',
                    'Soort melder': 'category',
                    'Melding hoe ontvangen': 'category',
                    'Eerste advies': 'category',
                    'Medewerker': 'category',
                    #'Hoofdprobleem (1 antwoord mogelijk)': 'category',
                    #'Reden afsluiting': 'category',
                    #'Voortgang proces door': 'category',
                    #'Soort instelling verwijzing': 'category',

                    'Aanwezigheid minderjarige kinderen': 'boolean',
                    'Is er sprake van huiselijk geweld?': 'boolean',
                    'Is er sprake van agressie / geweld?': 'boolean',
                    #'E33 melding': 'boolean',
                    'Wijk-GGD': 'boolean',
                    'DubbelRecord': 'boolean'
                    }

    # Cast columns
    df = df.astype(convert_dict)

    # Change date column Dtypes
    df[['Datum afsluiting',
        'Datum melding']] = df[['Datum afsluiting',
                                'Datum melding']].apply(pd.to_datetime)

    # Rename date columns for .apply function on multiple column names
    df.rename(columns={"Datum afsluiting": "Datum_afsluiting",
                    "Datum melding": "Datum_melding",
                    #"Besproken op datum": "Besproken_op_datum",
                    "Soort melder": "Soort_melder",
                    "Aanwezigheid minderjarige kinderen": "Aanwezigheid_minderjarige_kinderen",
                    "Is er sprake van huiselijk geweld?": "Is_er_sprake_van_huiselijk_geweld?",
                    "Is er sprake van agressie / geweld?": "Is_er_sprake_van_agressie_/_geweld?",
                    "Inhoud melding": "Inhoud_melding",
                    #"E33 melding": "E33_melding",
                    "Melding hoe ontvangen": "Melding_hoe_ontvangen",
                    "Vermoeden van problematiek": "Vermoeden_van_problematiek",
                    "Eerste advies": "Eerste_advies"}, inplace=True)
    return df