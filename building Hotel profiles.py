import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from functools import reduce 

"""
	Creating hotel profiles for each hotel.
	The final output is a csv file contain three columns of user, type and value. user identifies the name of each hotel, type indicates the features and value depends on feature indicates the value or weight of the feature. 
	---->	city which hotel is located
	---->   average price for each hotel
	---->   number of booking of each hotel
	---->   months of booking of each hotel
	---->   average commission of each hotel
	---->   4-top most used ratecodes
	---->   average length of stay per each hotel
	---->   average leads time per each hotel
"""		
header= ['Date','PROPERTY_ID','chainCode','rateCode','roomType','leadTime','nights','supplierCode','comissionEUR','officeID','company','city','Pricepernightroom','Month','year']
dataset = pd.read_csv('C:/previous laptop/iiamadeusNetwork/PolicyOutputNew_EndRP1.csv', sep=',', header=None, names=header , error_bad_lines=False, engine='python')


"""
	This part is for adding city for each hotel as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always city and value in the name of the city.
"""
citycode= dataset[['PROPERTY_ID','city']].drop_duplicates()
#citycode.to_csv('hotel-city.csv', index=False,header=False, sep=',')
citycodedf=pd.DataFrame()
citycodedf['user']=citycode['PROPERTY_ID']
citycodedf['type']='city'
citycodedf['value']=citycode['city']
#citycodedf.to_csv('hotel-feature-city.csv', index=False,header=False, sep=',')


"""
	This part is for adding average price for each hotel as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'AvgPrice' and value in teh average price of booking the hotel in general.
"""
pricemean=dataset.groupby(['PROPERTY_ID','Pricepernightroom']).size().reset_index(name='avg').sort_values(by='avg', ascending=False)
ndf = pricemean.assign(Mean = pricemean['Pricepernightroom'].abs()).groupby(['PROPERTY_ID'])\
     .agg({'Mean':'mean'})

ndf.to_csv('PROPERTY_ID_avgprice.csv', index=False,header=False, sep=',')
header= ['Mean']
ndf1 = pd.read_csv('PROPERTY_ID_avgprice.csv', sep=',', header=None, names=header, error_bad_lines=False, engine='python')
pricemean1=dataset.groupby(['PROPERTY_ID']).size().reset_index(name='count').sort_values(by='PROPERTY_ID', ascending=True)
Pricemeandf=pd.DataFrame()
Pricemeandf['user']=pricemean1['PROPERTY_ID']
Pricemeandf['type']='AvgPrice'
Pricemeandf['value']=ndf1['Mean']
#Pricemeandf.to_csv('PROPERTY_ID-feature-Avgprice.csv', index=False,header=False, sep=',')

"""
	This part is for adding number of booking of each hotel as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'Nbbooking' and value is the number of booking the hotel.
"""

NbBookingDF=pd.DataFrame()
NbBookingDF['user']=pricemean1['PROPERTY_ID']
NbBookingDF['type']='Nbbooking'
NbBookingDF['value']=pricemean1['count']
#NbBookingDF.to_csv('PROPERTY_ID-feature-Nbbooking.csv', index=False,header=False, sep=',')

"""
	This part is for adding months of booking of each hotel as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'Month' and value is the name of the month.
"""
month= dataset[['PROPERTY_ID','Month']]
#month.to_csv('PROPERTY_ID-Month.csv', index=False,header=False, sep=',')
Monthdf=pd.DataFrame()
Monthdf['user']=month['PROPERTY_ID']
Monthdf['type']='Month'
Monthdf['value']=month['Month']
#Monthdf.to_csv('PROPERTY_ID-feature-Month.csv', index=False,header=False, sep=',')

"""
	This part is for adding average commission of each hotel as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'AvgCommission' and value is the name of the month.
"""
commean=dataset.groupby(['PROPERTY_ID','comissionEUR']).size().reset_index(name='avg').sort_values(by='avg', ascending=False)
ndf2 = commean.assign(Mean = commean['comissionEUR'].abs()).groupby(['PROPERTY_ID'])\
     .agg({'Mean':'mean'})

#ndf2.to_csv('PROPERTY_ID_avgcom.csv', index=False,header=False, sep=',')
Commean1=dataset.groupby(['PROPERTY_ID']).size().reset_index(name='count').sort_values(by='PROPERTY_ID', ascending=True)
Commeandf=pd.DataFrame()
Commeandf['user']=Commean1['PROPERTY_ID']
Commeandf['type']='AvgCommission'
Commeandf['value']=ndf1['Mean']
#Commeandf.to_csv('PROPERTY_ID-feature-Avgcommission.csv', index=False,header=False, sep=',')

"""
	This part is for adding supplier code of each hotel as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'suppliercode' and value is the name of the month.
"""
suppliercode= dataset[['PROPERTY_ID','supplierCode']].drop_duplicates()
#suppliercode.to_csv('hotel-supplierCode.csv', index=False,header=False, sep=',')
suppliercodedf=pd.DataFrame()
suppliercodedf['user']=suppliercode['PROPERTY_ID']
suppliercodedf['type']='suppliercode'
suppliercodedf['value']=suppliercode['supplierCode']
#suppliercodedf.to_csv('PROPERTY_ID-feature-suppliercode.csv', index=False,header=False, sep=',')

ratecodeCoDF=dataset.groupby(['PROPERTY_ID','rateCode']).size().reset_index(name='count').sort_values(by='PROPERTY_ID', ascending=True)
"""
	just keep the 4 top ratecode per each hotel. Number 4 is the optimal number based on analysing data, but It can increase or decrease in future
	After that again, a dataframe is created which includes user, type, value columns for this feature.
	Here, type is the name of ratecode and value is the number of count of each ratecode per each hotel. I kept the count of each rate code, if we decide to give more weight to more frequent ratecodes.

"""
listrows=[]
patterndict={}
for index , row in ratecodeCoDF.iterrows():
    #print(row['company'])
    company=row['PROPERTY_ID']
    if row['PROPERTY_ID'] in patterndict:
        if patterndict[row['PROPERTY_ID']]==4:
            patterndict[row['PROPERTY_ID']]=4
        else:
            patterndict[row['PROPERTY_ID']]+=1
            row['count']=1
            listrows.append(row)
    else:
        patterndict[row['PROPERTY_ID']]=1
        row['count']=5
        listrows.append(row)
df = pd.DataFrame(listrows, columns=['PROPERTY_ID','rateCode','count'])
rateCode= dataset[['PROPERTY_ID','rateCode']]
rateCode.to_csv('PROPERTY_ID-rateCode1.csv', index=False,header=False, sep=',')
rateCodedf=pd.DataFrame()
rateCodedf['user']=df['PROPERTY_ID']
rateCodedf['type']=df['rateCode']
rateCodedf['value']=df['count']
#rateCodedf.to_csv('PROPERTY_ID-feature-rateCodeCount.csv', index=False,header=False, sep=',')

"""
	This part is for adding average length of stay per each hotel as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'NoNight' and value is average length of stay in each hotel.
"""

Nonightsavg=dataset.groupby(['PROPERTY_ID','nights']).size().reset_index(name='avg').sort_values(by='avg', ascending=False)
ndf = Nonightsavg.assign(Mean = Nonightsavg['nights'].abs()).groupby(['PROPERTY_ID'])\
     .agg({'Mean':'mean'})

ndf.to_csv('PROPERTY_ID_avgnoNights.csv', index=False,header=False, sep=',')
header= ['Mean']
ndf1 = pd.read_csv('PROPERTY_ID_avgnoNights.csv', sep=',', header=None, names=header, error_bad_lines=False, engine='python')
Nonightsavg1=dataset.groupby(['PROPERTY_ID']).size().reset_index(name='count').sort_values(by='PROPERTY_ID', ascending=True)
Nonightdf=pd.DataFrame()
Nonightdf['user']=Nonightsavg1['PROPERTY_ID']
Nonightdf['type']='NoNight'
Nonightdf['value']=ndf1['Mean']
#Nonightdf.to_csv('PROPERTY_ID-feature-Nonight.csv', index=False,header=False, sep=',')

"""
	This part is for adding average leads time per each hotel as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'AvgLeadTime' and value is average lead time for booking each hotel.
"""

leadsTavg=dataset.groupby(['PROPERTY_ID','leadTime']).size().reset_index(name='avg').sort_values(by='avg', ascending=False)
ndf = leadsTavg.assign(Mean = leadsTavg['leadTime'].abs()).groupby(['PROPERTY_ID'])\
     .agg({'Mean':'mean'})

ndf.to_csv('PROPERTY_ID_avgleadTime.csv', index=False,header=False, sep=',')
header= ['Mean']
ndf1 = pd.read_csv('PROPERTY_ID_avgleadTime.csv', sep=',', header=None, names=header, error_bad_lines=False, engine='python')
leadsTavg1=dataset.groupby(['PROPERTY_ID']).size().reset_index(name='count').sort_values(by='PROPERTY_ID', ascending=True)
LeadTimedf=pd.DataFrame()
LeadTimedf['user']=leadsTavg1['PROPERTY_ID']
LeadTimedf['type']='AvgLeadTime'
LeadTimedf['value']=ndf1['Mean']
#LeadTimedf.to_csv('PROPERTY_ID-feature-AvgLeadTime.csv', index=False,header=False, sep=',')

"""
	This part, I create a dataframe which is combination of all previous created dataframes and I save this final dataframe as a csv file.
"""
frames = [citycodedf,Pricemeandf,NbBookingDF,Commeandf,suppliercodedf,Monthdf,rateCodedf,Nonightdf,LeadTimedf]

result = pd.concat(frames)
result.to_csv('itemprofile.csv', index=False,header=True, sep=',')

"""
	we can add more features to this hotel profile later, if we decide to add sponsers or other information.
	e.g Chain code for each hotel can be added as well.
"""
