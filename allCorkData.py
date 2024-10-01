# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:54:18 2024

@author: prana
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingValues
import DBConnect



import warnings
warnings.filterwarnings('ignore')



# Example SQL query
sql_query = 'SELECT OrganisationName,schemename,populationserved,volumesupplied,yearofreturn,sampleid,sampledate,MonitoredLocationCode,schemecode,supplytypecode,parameterdescription,NVL(textresult, 0) AS textresult FROM dw_cork where yearofreturn between 2018 and 2022';

# Execute the query and fetch results into a DataFrame
data = pd.read_sql(sql_query, con=DBConnect.connection)

# Extract the month
data['MONTH'] = data['SAMPLEDATE'].dt.month

# convert datetime column to just date
data['SAMPLEDATE'] = pd.to_datetime(data['SAMPLEDATE']).dt.date

# Rename TEXTRESULT to RESULT
data.rename(columns={'TEXTRESULT': 'RESULT'}, inplace=True)

# to drop column 
data.drop('MONITOREDLOCATIONCODE',inplace=True,axis=1)

test_AllcorkData=data[(data['SAMPLEID'] == '2021/1350')]

## call function
missingValues.show_missing(data)

# Convert the column to a decimal data type using pd.to_numeric() with errors='coerce'
data['RESULT'] = pd.to_numeric(data['RESULT'], errors='coerce')


################################ EDA NITRATE ###########################

# box plot and violin plot for CC

df_CCounty = data[(data['PARAMETERDESCRIPTION'] == 'Nitrate')  & 
                  (data['ORGANISATIONNAME'] == 'Cork County Council')]


df_CCounty_nonZero=data[(data['PARAMETERDESCRIPTION'] == 'Nitrate') & 
                     (data['ORGANISATIONNAME'] == 'Cork County Council') &
                     (data['RESULT'] != 0.00)]




# histogram 
sns.histplot(df_CCounty['RESULT'], bins=60,kde=True, stat="density", color="skyblue");
plt.title('Overall Distribution of NITRATE');

# Create a distribution plot
sns.histplot(df_CCounty['RESULT'], kde=True, stat="density", color="skyblue")

# Overlay a vertical line at a specified position (e.g., mean)
plt.axvline(x=np.mean(df_CCounty['RESULT']), color='red', linestyle='--')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution Plot of Nitrate with Mean Line')

# Show the plot
plt.show()


# 
column_summary = df_CCounty['RESULT'].describe()

# Print or view the summary statistics
print(column_summary)


#df_CCounty.head(50)
#df_CCounty

df_CCounty.plot.box(column="RESULT", by="YEAROFRETURN", figsize=(10, 8))
sns.boxplot(x = 'YEAROFRETURN', y = 'RESULT', data = df_CCounty).set_title('Yearly Distribution of Nitrate-CorkCounty')
#Violinplot
#sns.violinplot(data=df_CCounty, x="YEAROFRETURN", y="RESULT")


# count of vlues 
# Group by 'category' column and count occurrences
grouped_NI = df_CCounty.groupby('RESULT').size().reset_index(name='count')

# Calculate percentage
total_count = grouped_NI['count'].sum()
grouped_NI['percentage'] = (grouped_NI['count'] / total_count) * 100

print(grouped_NI)

## co-relation of Nitrate with other chemicals 
c= df_CCounty.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c




################################ EDA IRON ###########################


df_CCounty_Iron = data[(data['PARAMETERDESCRIPTION'] == 'Iron')  & 
                  (data['ORGANISATIONNAME'] == 'Cork County Council')]


df_CCounty_nonZero_Iron =data[(data['PARAMETERDESCRIPTION'] == 'Iron') & 
                     (data['ORGANISATIONNAME'] == 'Cork County Council') &
                     (data['RESULT'] != 0.00)]

df_CCounty_Alter =data[(data['PARAMETERDESCRIPTION'] == 'Iron') & 
                     (data['ORGANISATIONNAME'] == 'Cork County Council') &
                     (data['RESULT']!= 20)]

df_CCounty_Alter2 =data[(data['PARAMETERDESCRIPTION'] == 'Iron') & 
                     (data['ORGANISATIONNAME'] == 'Cork County Council') &
                     (data['RESULT']> 0)& (data['RESULT']<= 200)]

df_CCounty_Alter.plot.box(column="RESULT", by="YEAROFRETURN", figsize=(10, 8))
sns.boxplot(x = 'YEAROFRETURN', y = 'RESULT', data = df_CCounty_Alter2).set_title('Yearly Distribution of Iron-CorkCounty')




## count of vlues 
# Group by 'category' column and count occurrences
grouped_fe = df_CCounty_Iron.groupby('RESULT').size().reset_index(name='count')

# Calculate percentage
total_count_fe = grouped_fe['count'].sum()
grouped_fe['percentage'] = (grouped_fe['count'] / total_count_fe) * 100

print(grouped_fe)


# histogram 
sns.histplot(df_CCounty_Iron['RESULT'], bins=60,kde=True, stat="density", color="skyblue");
plt.title('OverAll Distribution of Iron');



# Create a distribution plot
sns.histplot(df_CCounty_Alter2['RESULT'], kde=True, stat="density", color="skyblue")

# Overlay a vertical line at a specified position (e.g., mean)
plt.axvline(x=np.mean(df_CCounty_Alter2['RESULT']), color='red', linestyle='--')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Distribution Plot of Iron with Mean Line')

# Show the plot
plt.show()

# describe 
column_summary_Iron = df_CCounty_Alter['RESULT'].describe()

print(column_summary_Iron)



### yearly distribution of Iron #####


df_CCounty_Iron_2022 = data[(data['PARAMETERDESCRIPTION'] == 'Iron')  & 
                  (data['ORGANISATIONNAME'] == 'Cork County Council')& 
                  (data['YEAROFRETURN'] == 2022)]



sns.boxplot(x = 'YEAROFRETURN', y = 'RESULT', data = df_CCounty_Iron_2022).set_title('Yearly Distribution of Iron-CorkCounty')

unique_counts = df_CCounty_Iron_2022.groupby('RESULT')['RESULT'].nunique()

print(unique_counts)















