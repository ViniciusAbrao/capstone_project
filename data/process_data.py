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

transcript.columns - ['event', 'person', 'time', 'value']
event- ['offer received', 'offer viewed', 'transaction', 'offer completed']

portfolio - ['channels', 'difficulty', 'duration', 'id', 'offer_type', 'reward']

profile - ['age', 'became_member_on', 'gender', 'id', 'income']
   
Exploratory analysis - Taking a look to the raw data:

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
#Is'became_member_on' an important input feature to the model?

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


'''
Informational does not have completed offer, since it finishs with a transaction
inside the valid period.

11986 from 14825 persons completed offer. 
Are they influenced by the offer? 
Did the customer view the offer? 

Let's take the filtered data and apply the user_evaluate function to each user 
and to obtain the features and labels for the classification algorithm

'''

def user_evaluate(user):
    
    '''
    
os transcript de um dado usuario podem se enquadrar nos seguintes possiveis cenarios:   
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


      offer_type                                id      
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
  
diferentes perfis de usuarios (features), descritos em profile, sao avaliados para prever 
quais promocoes sao mais adequadas (labels)


    dado um usuario:
    
    loop nos eventos do usuario
    se evento = recebeu oferta 
        se oferta do tipo bogo/discount
            qual periodo da informational?
            pega data de recebimento e calcula data limite baseado no periodo
            loop nos eventos seguintes: 
                se usuario viu esta mesma oferta
                    loop nos enevtos seguintes:
                        se completou esta mesma oferta oferta dentro da data limite
                            y[i]=1
        se oferta do tipo informational
            qual periodo da informational?
            pega data de recebimento e calcula data limite baseado no periodo
            loop nos eventos seguintes
                se viu este mesmo informational
                    loop nos eventos seguintes
                        se comprou dentro da data limite
                            y[i]=1
    adiciona dados do usuario na lista de features
    adiciona y na lista de labels
                            
    '''
    
    teste=filter_transcript[filter_transcript['person']==user]\
            .loc[:,['event','time','value']].sort_values(by='time')         
    y_user=[0,0,0,0,0,0,0,0,0,0]
    cont=-1
    for events in list(teste['event']):
        cont=cont+1
        if events=='offer received':
            promo=list(teste['value'])[cont]['offer id']
            if promo in list_promos:
                duration=portfolio[portfolio['id']==promo]['duration']\
                    .values[0]*24 #now in hours
                received_at=list(teste['time'])[cont] #hours
                valid=received_at+duration
                cont_two=cont-1
                for events_two in (list(teste['event'])[cont:]):
                    cont_two=cont_two+1
                    if events_two=='offer viewed':
                        promo_two=list(teste['value'])[cont_two]['offer id']
                        if promo_two==promo:
                            cont_three=cont_two-1
                            for events_three in (list(teste['event'])[cont_two:]):
                                cont_three=cont_three+1
                                if events_three=='offer completed':
                                    promo_three=list(teste['value'])[cont_three]['offer_id']
                                    date_buy=list(teste['time'])[cont_three]
                                    if promo_three==promo and date_buy<=valid:
                                        indx=np.where(all_promos==promo)
                                        y_user[indx[0][0]]=1    
                                        break
            if promo in list_informational:
                duration=portfolio[portfolio['id']==promo]['duration']\
                    .values[0]*24 #now in hours
                received_at=list(teste['time'])[cont] #hours
                valid=received_at+duration
                cont_two=cont-1
                for events_two in (list(teste['event'])[cont:]):
                    cont_two=cont_two+1
                    if events_two=='offer viewed':
                        promo_two=list(teste['value'])[cont_two]['offer id']
                        if promo_two==promo:
                            cont_three=cont_two-1
                            for events_three in (list(teste['event'])[cont_two:]):
                                cont_three=cont_three+1
                                if events_three=='transaction': 
                                    date_buy=list(teste['time'])[cont_three]
                                    if date_buy<=valid:
                                        indx=np.where(all_promos==promo)
                                        y_user[indx[0][0]]=1    
                                        break
    y_train_test.append(np.array(y_user))
    x_train_test.append(np.array(profile[profile['id']==user]\
                                 .loc[:,['age','gender','income','became_member_on']]\
                                     .values[0]))


'''
Example of function results:

x_train_test=[]
y_train_test=[] 
user='78afa995795e4d85b5d9ceeca43f5fef' 
user_evaluate(user)


--- filter_transcript[filter_transcript['person']==user]\
            .loc[:,['event','time','value']].sort_values(by='time') 
                
Out[45]: 
                  event  ...                                              value
0        offer received  ...   {'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}
15561      offer viewed  ...   {'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}
47582       transaction  ...                                  {'amount': 19.89}
47583   offer completed  ...  {'offer_id': '9b98b8c7a33c4b65b9aebfe6a799e6d9...
49502       transaction  ...                                  {'amount': 17.78}
53176    offer received  ...   {'offer id': '5a8bc65990b245e5a138643cd4eb9837'}
85291      offer viewed  ...   {'offer id': '5a8bc65990b245e5a138643cd4eb9837'}
87134       transaction  ...                                  {'amount': 19.67}
92104       transaction  ...                                  {'amount': 29.72}
141566      transaction  ...                                  {'amount': 23.93}
150598   offer received  ...   {'offer id': 'ae264e3637204a6fb9bb56bc8210ddfd'}
163375     offer viewed  ...   {'offer id': 'ae264e3637204a6fb9bb56bc8210ddfd'}
201572   offer received  ...   {'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}
218393      transaction  ...                                  {'amount': 21.72}
218394  offer completed  ...  {'offer_id': 'ae264e3637204a6fb9bb56bc8210ddfd...
218395  offer completed  ...  {'offer_id': 'f19421c1d4aa40978ebb69ca19b0e20d...
230412      transaction  ...                                  {'amount': 26.56}
262138     offer viewed  ...   {'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}

[18 rows x 3 columns]
    
 
--- y_train_test
Out[44]: [array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])

      offer_type                                id      
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

--- more example users to test:
       '43fbc1418ee14268a5d3797006cc69be'
       '0610b486422d4921ae7d2bf64640c50b'
       '78afa995795e4d85b5d9ceeca43f5fef'
       'e2127556f4f64592b11af22de27a7932' 
       '01d26f638c274aa0b965d24cefe3183f'
       '9dc1421481194dcd9400aec7c9ae6366'
       'e4052622e5ba45a8b96b59aba68cf068'
  
''' 

list_promos=np.array(['ae264e3637204a6fb9bb56bc8210ddfd','4d5c57ea9a6940dd891ad53e9dbe8da0',\
           '9b98b8c7a33c4b65b9aebfe6a799e6d9','f19421c1d4aa40978ebb69ca19b0e20d',\
               '0b1e1539f2cc45b7b9fa7c272da2e1d7','2298d6c36e964ae4a3e7e9706d1fb8c2',\
                   'fafdcd668e3743c1bb461111dcafc2a4','2906b810c7d4411798c6938adc9daaa5',])
list_informational=np.array(['3f207df678b143eea3cee63160fa8bed',\
                             '5a8bc65990b245e5a138643cd4eb9837'])   
all_promos=np.array(['ae264e3637204a6fb9bb56bc8210ddfd','4d5c57ea9a6940dd891ad53e9dbe8da0',\
           '9b98b8c7a33c4b65b9aebfe6a799e6d9','f19421c1d4aa40978ebb69ca19b0e20d',\
               '0b1e1539f2cc45b7b9fa7c272da2e1d7','2298d6c36e964ae4a3e7e9706d1fb8c2',\
                   'fafdcd668e3743c1bb461111dcafc2a4','2906b810c7d4411798c6938adc9daaa5',\
                       '3f207df678b143eea3cee63160fa8bed',\
                           '5a8bc65990b245e5a138643cd4eb9837'])

#mapping 'gender' from str to int, in order to use it in the classifier
# F -> 1
# M -> 2
# O -> 3
profile['gender']=profile['gender'].map(lambda x:1 if x =='F' else (2 if x=='M' else 3))

x_train_test=[]
y_train_test=[]  

import time
import progressbar
with progressbar.ProgressBar(max_value=len(list(filter_users))) as bar:
    contbar=0
    for user in list(filter_users):    
        user_evaluate(user)
        bar.update(contbar)
        contbar=contbar+1

'''
Saving the data

''' 
import pickle
with open('features.pkl', 'wb') as file:
    pickle.dump(x_train_test, file)
    
with open('labels.pkl', 'wb') as file:
    pickle.dump(y_train_test, file)

'''
loading the data to check

'''

with open('features.pkl', 'rb') as file:
    features = pickle.load(file)

with open('features.pkl', 'rb') as file:
    labels = pickle.load(file)
        



