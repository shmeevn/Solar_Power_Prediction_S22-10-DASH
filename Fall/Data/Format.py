import pandas as pd

df = pd.read_csv('PVdata.csv', sep=',')
df = df.drop(['PCS_AC_Power', 'Battery_Power', 'Microgrid_Total_Load', 'Microgrid_Net_Power'], axis = 1)

df[['Date', 'Time']] = df['Time'].str.split(', ', 1, expand=True)
last_col = df.pop('Date')
df.insert(loc = 0, column = 'Date', value = last_col)
df['Time'] = df['Time'].map(lambda x: x.rstrip(' CDT'))

tdf = []
selector = 0 # if 0, then the output file will have categorical times. if 1, the output file will have times in 24hr format

for i in df['Time']:
    if 'AM' in i:
        i = i.rstrip(' AM')
        i = i.replace('12', '0')
    if 'PM' in i:
        i = i.rstrip(' PM')
        if i[0:2] != '12':
            temp = int(i[0:2].rstrip(':')) + 12
            if len(i[0:2].rstrip(':')) < 2:
                i = str(temp) + ':' + i[2:]
            else:
                i = str(temp) + i[2:]
    if selector == 0:
        i = i.replace(':', '.')
        i = i.rstrip('.00')
        if i == '':
            i = '0.0'
    tdf.append(i)

tempdf = pd.DataFrame(tdf, columns = ['Time'])
df['Time'] = tempdf

print(df['Time'])

df.to_csv('Formatted_PV_data.csv', sep=';', index = False)