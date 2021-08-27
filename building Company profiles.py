import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from functools import reduce
"""
	Creating Company profiles for each company.
	The final output is a csv file contain three columns of user, type and value. user identifies the name of each company, type indicates the features and value depends on feature indicate the value or weight of the feature. 
	---->	4 top visited city per each company
	---->   the number of cities in general per each company
	---->   4 top most used rate code per each company
	---->   officeId for each company
	---->   average budget for booking per each company
	---->   average length of stay 
	---->   number of booking per each comapny
	---->   months of booking of each comapny
"""
header= ['Date','PROPERTY_ID','chainCode','rateCode','roomType','leadTime','nights','supplierCode','comissionEUR','officeID','company','city','Pricepernightroom','Month','year']
dataset = pd.read_csv('C:/previous laptop/iiamadeusNetwork/PolicyOutputNew_EndRP1.csv', sep=',', header=None,names=header , error_bad_lines=False, engine='python')


"""
	just keep the 4 top visited city per each company. Number 4 is the optimal number based on analysing data, but It can increase or decrease in future
	After that again, a dataframe is created which includes user, type, value columns for this feature.
	Here, type is the always 'city' and value is the name of the city each. 
	Also, I'm keeping the number of cities which each company traveled to them in general, I keep the ceiling value of root cube of real average, to have a fixed number of categories for nb of cities.
"""
citypercompany=dataset.groupby(['company','city']).size().reset_index(name='count').sort_values(by=['company','count'], ascending=False)
#citypercompany.to_csv('company-city-count.csv', index=False,header=True, sep=',')
citypercompanyDF = pd.read_csv('company-city-count.csv', sep=',', header=0 , error_bad_lines=False, engine='python')
listrowscity=[]
patterndict={}
for index , row in citypercompanyDF.iterrows():
    #print(row['company'])
    company=row['company']
    if row['company'] in patterndict:
        if patterndict[row['company']]==4:
            patterndict[row['company']]=4
        else:
            patterndict[row['company']]+=1
            row['weight']=1
            listrowscity.append(row)
    else:
        patterndict[row['company']]=1
        row['weight']=1
        listrowscity.append(row)

dfcity = pd.DataFrame(listrowscity, columns=['company','city','count','weight'])
Citydf=pd.DataFrame()
Citydf['user']=dfcity['company']
Citydf['type']=dfcity['city']
Citydf['value']=dfcity['weight']
#Citydf.to_csv('company-feature-eachCity.csv', index=False,header=False, sep=',')
companyCityNB=citypercompanyDF.groupby(['company']).size().reset_index(name='count').sort_values(by=['count'], ascending=False)
#companyCityNB.to_csv('company-city-NB.csv', index=False,header=True, sep=',')

import math
listrowsNB=[]
citycompanyNBDF = pd.read_csv('company-city-NB.csv', sep=',', header=0 , error_bad_lines=False, engine='python')
for index , row in citycompanyNBDF.iterrows():
    cnt=float(row['count'])
    row['count']=math.ceil((cnt)**(1/3))
    listrowsNB.append(row)
citycompanydf1 = pd.DataFrame(listrowsNB, columns=['company','count'])
NBCitydf=pd.DataFrame()
NBCitydf['user']=citycompanydf1['company']
NBCitydf['type']='Nb_city'
NBCitydf['value']=citycompanydf1['count']
#NBCitydf.to_csv('company-feature-nbcity.csv', index=False,header=False, sep=',')

citycode= dataset[['company','city']]
citycode.to_csv('company-city.csv', index=False,header=False, sep=',')
citycodedf=pd.DataFrame()
citycodedf['user']=citycode['company']
citycodedf['type']='city'
citycodedf['value']=citycode['city']
#citycodedf.to_csv('company-feature-city.csv', index=False,header=False, sep=',')


"""
	just keep the 4 top rate code per each company. Number 4 is the optimal number based on analysing data, but It can increase or decrease in future
	After that again, a dataframe is created which includes user, type, value columns for this feature.
	Here, type is the name of ratecode and value is the number of count of each ratecode per each company. I kept the count of each rate code, if we decide to give more weight to more frequent ratecodes.
	
"""
countpercompany=dataset.groupby(['company','rateCode']).size().reset_index(name='count').sort_values(by=['company','count'], ascending=False)
#countpercompany.to_csv('company-ratecode-count.csv', index=False,header=True, sep=',')
ratecodeCoDF = pd.read_csv('company-ratecode-count.csv', sep=',', header=0 , error_bad_lines=False, engine='python')


listrows=[]
patterndict={}
for index , row in ratecodeCoDF.iterrows():
    #print(row['company'])
    company=row['company']
    if row['company'] in patterndict:
        if patterndict[row['company']]==4:
            patterndict[row['company']]=4
        else:
            patterndict[row['company']]+=1
            row['count']=1
            listrows.append(row)
    else:
        patterndict[row['company']]=1
        row['count']=5
        listrows.append(row)

rateCode= dataset[['company','rateCode']]
#rateCode.to_csv('company-rateCode1.csv', index=False,header=False, sep=',')
rateCodedf=pd.DataFrame()
rateCodedf['user']=df['company']
rateCodedf['type']=df['rateCode']
rateCodedf['value']=df['weight']
#rateCodedf.to_csv('company-feature-rateCodeCount.csv', index=False,header=False, sep=',')

"""
	This part is for adding officeId for each company as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'officeID' and value is name of the officeID.
	I did not add the officeId in final dataframe, but initially I just created.
	
"""
officeID= dataset[['company','officeID']]
#officeID.to_csv('company-officeID1.csv', index=False,header=False, sep=',')
officeIDdf=pd.DataFrame()
officeIDdf['user']=officeID['company']
officeIDdf['type']='officeID'
officeIDdf['value']=officeID['officeID']
#officeIDdf.to_csv('company-feature-officeID.csv', index=False,header=False, sep=',')


"""
	This part is for adding average price each company spend for booking hotel in general as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'AvgPrice' and value in the average price of booking the hotel in general.
	
"""
pricemean=dataset.groupby(['company','Pricepernightroom']).size().reset_index(name='avg').sort_values(by='avg', ascending=False)
ndf = pricemean.assign(Mean = pricemean['Pricepernightroom'].abs()).groupby(['company'])\
     .agg({'Mean':'mean'})

ndf.to_csv('Company_avgprice.csv', index=False,header=False, sep=',')
header= ['Mean']
ndf1 = pd.read_csv('Company_avgprice.csv', sep=',', header=None, names=header, error_bad_lines=False, engine='python')
pricemean1=dataset.groupby(['company']).size().reset_index(name='count').sort_values(by='company', ascending=True)
Pricemeandf=pd.DataFrame()
Pricemeandf['user']=pricemean1['company']
Pricemeandf['type']='AvgPrice'
Pricemeandf['value']=ndf1['Mean']
#Pricemeandf.to_csv('company-feature-Avgprice.csv', index=False,header=False, sep=',')

"""
	This part is for adding number of booking per each comapny in general as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'Nbbooking' and value is the number of booking of hotels per each company.
"""

NbBookingDF=pd.DataFrame()
NbBookingDF['user']=pricemean1['company']
NbBookingDF['type']='Nbbooking'
NbBookingDF['value']=pricemean1['count']
#NbBookingDF.to_csv('company-feature-Nbbooking.csv', index=False,header=False, sep=',')

"""
	This part is for adding months of booking per each company as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'Month' and value is the name of the month wich company had a booking.
"""
month= dataset[['company','Month']]
#month.to_csv('company-Month.csv', index=False,header=False, sep=',')
Monthdf=pd.DataFrame()
Monthdf['user']=month['company']
Monthdf['type']='Month'
Monthdf['value']=month['Month']
#Monthdf.to_csv('company-feature-Month.csv', index=False,header=False, sep=',')


"""
	This part is for adding average length of stay per each comapny as one main feature in their profiles.
	I'm creating a dataframe includes of user, type, value columns for this feature.
	Here, type is always 'NoNight' and value is average length of stay per each comapny.
"""
Nonightsavg=dataset.groupby(['company','nights']).size().reset_index(name='avg').sort_values(by='avg', ascending=False)
ndf = Nonightsavg.assign(Mean = Nonightsavg['nights'].abs()).groupby(['company'])\
     .agg({'Mean':'mean'})

#ndf.to_csv('company_avgnoNights.csv', index=False,header=False, sep=',')

header= ['Mean']
ndf1 = pd.read_csv('company_avgnoNights.csv', sep=',', header=None, names=header, error_bad_lines=False, engine='python')
Nonightsavg1=dataset.groupby(['company']).size().reset_index(name='count').sort_values(by='company', ascending=True)
Nonightdf=pd.DataFrame()
Nonightdf['user']=Nonightsavg1['company']
Nonightdf['type']='NoNight'
Nonightdf['value']=ndf1['Mean']
#Nonightdf.to_csv('Company-feature-Nonight.csv', index=False,header=False, sep=',')

"""
	This part, I create a dataframe which is combination of all previous created dataframes and I save this final dataframe as a csv file.
"""
frames = [Citydf,NBCitydf, rateCodedf,Pricemeandf,NbBookingDF,Monthdf,Nonightdf]

result = pd.concat(frames)
result.to_csv('Userprofile.csv', index=False,header=True, sep=',')

