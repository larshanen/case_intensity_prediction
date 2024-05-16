import pandas as pd
import numpy as np

# Store all Dutch national holidays from 2019 to 2025
holidays = ['2019-01-01', '2019-04-19', '2019-04-21', '2019-04-22', '2019-04-27', '2019-05-05', '2019-05-30', '2019-06-09', '2019-12-25', '2019-12-26',
            '2020-01-01', '2020-04-10', '2020-04-12', '2020-04-13', '2020-04-27', '2020-05-05', '2020-05-21', '2020-05-31', '2020-12-25', '2020-12-26',
            '2021-01-01', '2021-04-02', '2021-04-04', '2021-04-05', '2021-04-27', '2021-05-05', '2021-05-13', '2021-05-23', '2021-12-25', '2021-12-26',
            '2022-01-01', '2022-04-15', '2022-04-17', '2022-04-18', '2022-04-27', '2022-05-05', '2022-05-26', '2022-06-05', '2022-12-25', '2022-12-26',
            '2023-01-01', '2023-04-07', '2023-04-09', '2023-04-10', '2023-04-27', '2023-05-05', '2023-05-18', '2023-05-28', '2023-12-25', '2023-12-26',
            '2024-01-01', '2024-03-29', '2024-03-31', '2024-04-01', '2024-04-27', '2024-05-05', '2024-05-09', '2024-05-19', '2024-12-25', '2024-12-26',
            '2025-01-01', '2025-04-18', '2025-04-20', '2025-04-21', '2025-04-27', '2025-05-05', '2025-05-29', '2025-06-08', '2025-12-25', '2025-12-26']

# Function to convert string of dates to list of datetime objects
def days_extractor(date_notification, dates_interventions, freq='C'):
    if dates_interventions == []:
        return 0
    else:
        werkdagen = pd.bdate_range(date_notification, max(dates_interventions), inclusive='right', freq=freq, holidays=holidays).shape[0]
        return int(werkdagen)

# Define the function to calculate the amount of interventions
def interventions_extractor(dates):
    if not dates:
        return 0
    else:
        return len(dates)