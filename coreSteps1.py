import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from missingValues import show_missing
import DBConnect




# Example SQL query
sql_query3 = 'select * from CorkCountyParamView'

# Execute the query and fetch results into a DataFrame
df = pd.read_sql(sql_query3, con=DBConnect.connection)

# Close the connection
#connection.close()


df.head(50)

df.shape[0]

# View the datatypes 
df.info()

# to drop column 
df.drop('TETRA_TRI_CHLOROETHENE',inplace=True,axis=1)
df.drop('COLONY_COUNT',inplace=True,axis=1)


#To convert object datatype to Integer datatype
#df.SAMPLEDATE=df.SAMPLEDATE.astype("date")

# convert datetime column to just date
df['SAMPLEDATE'] =  pd.to_datetime(df['SAMPLEDATE'], errors='coerce')

df['MONTH'] = df['SAMPLEDATE'].dt.month



# function to print dtypes, count unique and missing values.
show_missing(df)


    

# Get all the parameter columns into variable 
cols2 = ['DICHLOROETHANE',
'TRICHLOROBENZOIC_ACID',
'TWOFOURD',
'ALUMINIUM',
'AMMONIUM',
'ANTIMONY',
'ARSENIC',
'ATRAZINE',
'BENTAZONE',
'BENZENE',
'BENZO_A_PYRENE',
'BORON',
'BROMATE',
'CADMIUM',
'CHLORFENVINPHOS',
'CHLORIDE',
'CHROMIUM',
'CLOPYRALID',
'CLOSTRIDIUM_PERFRINGENS',
'COLIFORM_BACTERIA',
'COLOUR',
'CONDUCTIVITY',
'COPPER',
'CRYPTOSPORIDIUM',
'CYANIDE',
'CYPERMETHRIN',
'DICHLOBENIL',
'DICHLORPROP',
'DIFLUFENICAN',
'DIURON',
'E_COLI',
'ENTEROCOCCI',
'FLUORIDE',
'FREE_CHLORINE',
'GIARDIA',
'GLYPHOSATE',
'IRON',
'ISOPROTURON',
'LEAD',
'LINURON',
'MCPA',
'MANGANESE',
'MECOPROP',
'MERCURY',
'METALDEHYDE',
'NICKEL',
'NITRATE',
'NITRITE_AT_TAP',
'ODOUR',
'PAH',
'PENDIMETHALIN',
'PESTICIDES_TOTAL',
'PROPYZAMIDE',
'SELENIUM',
'SIMAZINE',
'SODIUM',
'SULPHATE',
'TASTE',
'TOTAL_CHLORINE',
'TOTAL_ORGANIC_CARBON',
'TRICLOPYR',
'TRIHALOMETHANES_TOTAL',
'TURBIDITY_AT_TAP',
'PH']
    

# Convert 'None' values to NaN
df.replace('None', pd.NA, inplace=True)

# Replace NA values with 0
df.fillna(0, inplace=True)

# Iterate through all columns and change data type to float
for column in cols2:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    
    
# test samples 
df[(df['SAMPLEID'] == '2019/0680')]



# check the co-relation 
col2=df.loc[:,
['DICHLOROETHANE',
'TRICHLOROBENZOIC_ACID',
'TWOFOURD',
'ALUMINIUM',
'AMMONIUM',
'ANTIMONY',
'ARSENIC',
'ATRAZINE',
'BENTAZONE',
'BENZENE',
'BENZO_A_PYRENE',
'BORON',
'BROMATE',
'CADMIUM',
'CHLORFENVINPHOS',
'CHLORIDE',
'CHROMIUM',
'CLOPYRALID',
'COLIFORM_BACTERIA',
'COLOUR',
'CONDUCTIVITY',
'COPPER',
'CRYPTOSPORIDIUM',
'CYANIDE',
'CYPERMETHRIN',
'DICHLOBENIL',
'DICHLORPROP',
'DIFLUFENICAN',
'DIURON',
'E_COLI',
'ENTEROCOCCI',
'FLUORIDE',
'FREE_CHLORINE',
'GIARDIA',
'GLYPHOSATE',
'IRON',
'ISOPROTURON',
'LEAD',
'LINURON',
'MCPA',
'MANGANESE',
'MECOPROP',
'MERCURY',
'METALDEHYDE',
'NICKEL',
'NITRATE',
'NITRITE_AT_TAP',
'PAH',
'PENDIMETHALIN',
'PESTICIDES_TOTAL',
'PROPYZAMIDE',
'SELENIUM',
'SIMAZINE',
'SODIUM',
'SULPHATE',
'TOTAL_CHLORINE',
'TOTAL_ORGANIC_CARBON',
'TRICLOPYR',
'TRIHALOMETHANES_TOTAL',
'TURBIDITY_AT_TAP',
'PH',
]]

corr = col2.corr(method='kendall')
corr.style.background_gradient(cmap='coolwarm').set_precision(2)




c= col2.corr()
sns.heatmap(c,cmap="BrBG",annot=True)

# co-relation of nitrate only 
cor_nitrate=c['NITRATE']

# co-relation of Iron only
cor_Iron=c['IRON']


### dataset for Nitrate #####

# check the co-relation 
col2_nitrate=df.loc[:,
['DICHLOROETHANE',
'TRICHLOROBENZOIC_ACID',
'TWOFOURD',
'ALUMINIUM',
'AMMONIUM',
'ANTIMONY',
'ARSENIC',
'ATRAZINE',
'BENTAZONE',
'BENZENE',
'BENZO_A_PYRENE',
'BORON',
'BROMATE',
'CADMIUM',
'CHLORFENVINPHOS',
'CHLORIDE',
'CHROMIUM',
'CLOPYRALID',
'CONDUCTIVITY',
'COPPER',
'CRYPTOSPORIDIUM',
'CYANIDE',
'CYPERMETHRIN',
'DICHLOBENIL',
'DICHLORPROP',
'DIFLUFENICAN',
'NITRITE_AT_TAP',
'DIURON',
'FLUORIDE',
'GLYPHOSATE',
'ISOPROTURON',
'LEAD',
'LINURON',
'MCPA',
'MECOPROP',
'MERCURY',
'METALDEHYDE',
'NICKEL',
'NITRATE',
'PAH',
'PENDIMETHALIN',
'PESTICIDES_TOTAL',
'PROPYZAMIDE',
'SELENIUM',
'SIMAZINE',
'SODIUM',
'SULPHATE',
'TOTAL_ORGANIC_CARBON',
'TRICLOPYR',
'TRIHALOMETHANES_TOTAL',
'TURBIDITY_AT_TAP',
'PH',
'MONTH'
]]





col2_nitrate4=df.loc[:,
['PAH',
'BENTAZONE',
'NITRITE_AT_TAP',
'NITRATE',
'SULPHATE'
]]

# df where nitrate is >0
df2 = col2_nitrate4[(col2_nitrate4['NITRATE'] >0)]










