from datetime import datetime

# Function to convert string of dates to list of datetime objects
def dates_converter(date_string):
    # Return empty list if no intervention dates are registered
    if not date_string:
        return []
    # Split by comma's and add to list
    else:
        date_list = date_string.split(', ')
        date_list = [datetime.strptime(date, '%Y-%m-%d').date() for date in date_list]
        return date_list
    
# Function to map municipalities
def municipality_mapper(string):
    WB_regions = ['Alphen-Chaam',
                  'Altena',
                  'Baarle-Nassau',
                  'Bergen op Zoom',
                  'Breda',
                  'Drimmelen',
                  'Etten-Leur',
                  'Geertruidenberg',
                  'Halderberge',
                  'Moerdijk',
                  'Oosterhout',
                  'Roosendaal',
                  'Rucphen',
                  'Steenbergen',
                  'Woensdrecht',
                  'Zundert']
    Altena_regions = ['Aalburg',
                      'Werkendam',
                      'Woudrichem']
    
    if string in Altena_regions:
        return 'Altena'
    elif string not in WB_regions:
        return 'outside_WB'
    else:
        return string