# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:39:03 2020

@author: vinic_000
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# read in the json files
portfolio = pd.read_json('portfolio.json', orient='records', lines=True)
profile = pd.read_json('profile.json', orient='records', lines=True)
transcript = pd.read_json('transcript.json', orient='records', lines=True)

'''
primeiro dividir os casos de transcript em possiveis cenarios:
transcript - ['event', 'person', 'time', 'value']
transcript.columns - ['offer received', 'offer viewed', 'transaction', 'offer completed']
   
1-recebeu oferta (bogo or discount), viu, comprou dentro prazo, completou oferta : y[i]=1
2-recebeu oferta (informational), viu, comprou dentro prazo : y[i]=1
3-recebeu oferta, viu, comprou (dentro do prazo), nao completou oferta : y[:]=0
4-recebeu oferta, viu, comprou (fora do prazo), nao completou oferta : y[:]=0
5-recebeu oferta, viu, nao comprou, nao completou oferta : y[:]=0
6-recebeu oferta, nao viu, comprou, completou oferta : y[:]=0
7-recebeu oferta, nao viu, comprou, nao completou oferta : y[:]=0
8-recebeu oferta, nao viu, nao comprou, nao completou oferta : y[:]=0
9-demais casos : y[:]=0
    
y[i]=1 : i depende de qual das 10 possiveis ofertas (do portfolio) que foi enviada:
portfolio - ['channels', 'difficulty', 'duration', 'id', 'offer_type', 'reward']

      offer_type                                id      cases_of_offer_completed
y[0]           bogo  ae264e3637204a6fb9bb56bc8210ddfd   3657
y[1]           bogo  4d5c57ea9a6940dd891ad53e9dbe8da0   3310
y[2]           bogo  9b98b8c7a33c4b65b9aebfe6a799e6d9   4188
y[3]           bogo  f19421c1d4aa40978ebb69ca19b0e20d   4103
y[4]       discount  0b1e1539f2cc45b7b9fa7c272da2e1d7   3386
y[5]       discount  2298d6c36e964ae4a3e7e9706d1fb8c2   4886
y[6]       discount  fafdcd668e3743c1bb461111dcafc2a4   5003
y[7]       discount  2906b810c7d4411798c6938adc9daaa5   3911
y[8]  informational  3f207df678b143eea3cee63160fa8bed   0
y[9]  informational  5a8bc65990b245e5a138643cd4eb9837   0 
  
diferentes perfis de usuarios, descritos em profile, sao avaliados na promocao 
profile - ['age', 'became_member_on', 'gender', 'id', 'income']

para cada usuario olhar suas transcripts
Exemplo: user_transcripts=filter_transcript[filter_transcript['person']==\
    '109c657e856f4722b46a9183aaaf8c95']\
    .loc[:,['event','time','value']].sort_values(by='time')

percorrer user_transcripts encontrada para cada usuario:
- Se recebeu oferta bogo ou discount e atende cenario 1:
Inputs (age, gender, income) - Outputs (y[i]=1, i=0,1,2,3,4,5,6 ou 7)
- Se recebeu oferta information e atende cenario 2:
Inputs (age, gender, income) - Outputs (y[i]=1, i=8 ou 9)
- Se nao:
Inputs (age, gender, income) - Outputs (y[:]=0)    
   
'''

#histogram of user's age
plt.hist(profile['age'],bins=np.arange(profile['age'].max()+5))
no_info=profile[profile['age']==118].count()
print('how many users with no info')
print(no_info)
#2175 users with 118 years old, and no information of gender and no income
print(profile.count())

#lets filter the profile data frame
profile=profile[profile['age']!=118]
print('profile shape')
print(profile.shape)
plt.figure()
plt.hist(profile['age'],bins=np.arange(profile['age'].max()+5))

#taking a look at became_member_on columns
#how many different values?
member_since=len(profile['became_member_on'].unique())
print('how many different values of became_member_on?')
print(member_since)
#looks like some users became member on the same day
same_day_count=profile.groupby(by='became_member_on').count().sort_values(by='age')
print('became member on the same day?')
print(same_day_count)
print('users that became member on 20170819')
print(profile[profile['became_member_on']==20170819])
#looks like 'became_member_on' is not an important input feature to the model

#lets filter the transcript data frame to 14825 users with info
filter_users=profile['id'].unique()
filter_transcript=transcript[transcript['person'].isin(filter_users)]
print('percentage of data that have been filtered')
print((transcript.shape[0]-filter_transcript.shape[0])/transcript.shape[0])
print('how many transcripts remain')
print(transcript.count())

#extract id from transcript['value'] in case of 'offer completed'
event_completed=filter_transcript[filter_transcript['event']=='offer completed']\
    .loc[:,['event','time','value','person']]
event_completed_values=list(event_completed['value'].values)
extract_completed=(lambda x:x['offer_id'])
ids=pd.Series(pd.Series(event_completed_values).map(extract_completed))

print('how many cases of offer_completed for each offer_type')
print(len(ids[ids=='ae264e3637204a6fb9bb56bc8210ddfd']))
print(len(ids[ids=='4d5c57ea9a6940dd891ad53e9dbe8da0']))
print(len(ids[ids=='3f207df678b143eea3cee63160fa8bed']))
print(len(ids[ids=='9b98b8c7a33c4b65b9aebfe6a799e6d9']))
print(len(ids[ids=='0b1e1539f2cc45b7b9fa7c272da2e1d7']))
print(len(ids[ids=='2298d6c36e964ae4a3e7e9706d1fb8c2']))
print(len(ids[ids=='fafdcd668e3743c1bb461111dcafc2a4']))
print(len(ids[ids=='5a8bc65990b245e5a138643cd4eb9837']))
print(len(ids[ids=='f19421c1d4aa40978ebb69ca19b0e20d']))
print(len(ids[ids=='2906b810c7d4411798c6938adc9daaa5']))

print('how many differente persons have offer_completed')
print(event_completed['person'].nunique())
#11986 from 14825 persons completed offer. Are they influenced by the offer? 
#Did the customer view the offer? 

list_promos=np.array(['ae264e3637204a6fb9bb56bc8210ddfd','4d5c57ea9a6940dd891ad53e9dbe8da0',\
           '9b98b8c7a33c4b65b9aebfe6a799e6d9','f19421c1d4aa40978ebb69ca19b0e20d',\
               '0b1e1539f2cc45b7b9fa7c272da2e1d7','2298d6c36e964ae4a3e7e9706d1fb8c2',\
                   'fafdcd668e3743c1bb461111dcafc2a4','2906b810c7d4411798c6938adc9daaa5'])

x_train_test=[]
y_train_test=[]  

import time
import progressbar

with progressbar.ProgressBar(max_value=len(list(filter_users))) as bar:
    contbar=0
    for user in list(filter_users):    
        user_transcripts=filter_transcript[filter_transcript['person']==user]\
            .loc[:,['event','time','value']].sort_values(by='time')
        y_user=[0,0,0,0,0,0,0,0,0,0]
        cont=-1
        for events in list(user_transcripts['event']):
            cont=cont+1
            if events=='offer completed':
                promo=list(user_transcripts['value'])[cont]['offer_id']
                indx=np.where(list_promos==promo)
                y_user[indx[0][0]]=1    
        y_train_test.append(np.array(y_user))
        x_train_test.append(np.array(profile[profile['id']==user]\
                                     .loc[:,['age','gender','income']].values[0]))
        bar.update(contbar)
        contbar=contbar+1

 
    
'''
Exemplo 
user='43fbc1418ee14268a5d3797006cc69be'

                  event  ...                                              value
11323    offer received  ...   {'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}
59231    offer received  ...   {'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}
68569      offer viewed  ...   {'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}
82989       transaction  ...                                  {'amount': 21.32}
82990   offer completed  ...  {'offer_id': '0b1e1539f2cc45b7b9fa7c272da2e1d7...
82991   offer completed  ...  {'offer_id': '4d5c57ea9a6940dd891ad53e9dbe8da0...
94775       transaction  ...                                  {'amount': 26.95}
120691   offer received  ...   {'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}
126371     offer viewed  ...   {'offer id': '4d5c57ea9a6940dd891ad53e9dbe8da0'}
134664      transaction  ...                                  {'amount': 19.55}
134665  offer completed  ...  {'offer_id': '4d5c57ea9a6940dd891ad53e9dbe8da0...
136956      transaction  ...                                  {'amount': 21.13}
139078      transaction  ...                                  {'amount': 19.86}
160500   offer received  ...   {'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}
166690      transaction  ...                                  {'amount': 14.74}
166691  offer completed  ...  {'offer_id': 'f19421c1d4aa40978ebb69ca19b0e20d...
176627     offer viewed  ...   {'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}
181964      transaction  ...                                  {'amount': 39.66}
211408   offer received  ...   {'offer id': 'ae264e3637204a6fb9bb56bc8210ddfd'}
229831      transaction  ...                                  {'amount': 15.23}
229832  offer completed  ...  {'offer_id': 'ae264e3637204a6fb9bb56bc8210ddfd...
254993   offer received  ...   {'offer id': '0b1e1539f2cc45b7b9fa7c272da2e1d7'}
278924      transaction  ...                                  {'amount': 15.07}
281285      transaction  ...                                  {'amount': 29.47}
281286  offer completed  ...  {'offer_id': '0b1e1539f2cc45b7b9fa7c272da2e1d7...
    

y[0]           bogo  ae264e3637204a6fb9bb56bc8210ddfd   
y[1]           bogo  4d5c57ea9a6940dd891ad53e9dbe8da0   
y[2]           bogo  9b98b8c7a33c4b65b9aebfe6a799e6d9   
y[3]           bogo  f19421c1d4aa40978ebb69ca19b0e20d   
y[4]       discount  0b1e1539f2cc45b7b9fa7c272da2e1d7   
y[5]       discount  2298d6c36e964ae4a3e7e9706d1fb8c2   
y[6]       discount  fafdcd668e3743c1bb461111dcafc2a4   
y[7]       discount  2906b810c7d4411798c6938adc9daaa5   
y[8]  informational  3f207df678b143eea3cee63160fa8bed   
y[9]  informational  5a8bc65990b245e5a138643cd4eb9837   


user='43fbc1418ee14268a5d3797006cc69be'  
teste=filter_transcript[filter_transcript['person']==user]\
        .loc[:,['event','time','value']].sort_values(by='time')  
   
x_train_test=[]
y_train_test=[]  
    
y_user=[0,0,0,0,0,0,0,0,0,0]
cont=-1
for events in list(teste['event']):
    cont=cont+1
    if events=='offer completed':
        print(cont)
        promo=list(teste['value'])[cont]['offer_id']
        print(promo)
        indx=np.where(list_promos==promo)
        y_user[indx[0][0]]=1    
        print(y_user)
y_train_test.append(np.array(y_user))
x_train_test.append(np.array(profile[profile['id']==user]\
                             .loc[:,['age','gender','income']].values[0]))

    
na vdd tem que ser 

se usuario viu oferta do tipo bogo/discount:
    se depois completou
        y=1
        
se usuario recebeu oferta do tipo informational
    qual periodo da promocional?
    pega data de recebimento e calcula data limite baseado no periodo
    se viu informational
        se comprou dentro da data limite
            y=1

    
 '''   
    
    
user='43fbc1418ee14268a5d3797006cc69be'  
teste=filter_transcript[filter_transcript['person']==user]\
        .loc[:,['event','time','value']].sort_values(by='time')  
   
x_train_test=[]
y_train_test=[]  
    
y_user=[0,0,0,0,0,0,0,0,0,0]
cont=-1
for events in list(teste['event']):
    cont=cont+1
    if events=='offer completed':
        print(cont)
        promo=list(teste['value'])[cont]['offer_id']
        print(promo)
        indx=np.where(list_promos==promo)
        y_user[indx[0][0]]=1    
        print(y_user)
        
y_train_test.append(np.array(y_user))
x_train_test.append(np.array(profile[profile['id']==user]\
                             .loc[:,['age','gender','income']].values[0]))



