# Starbuck's Capstone Challenge


## Project overview (according to Udacity.com)
The program used to create the data simulates how people make purchasing decisions and how those decisions are influenced by promotional offers.
Each person in the simulation has some hidden traits that influence their purchasing patterns and are associated with their observable traits. People produce various events, including receiving offers, opening offers, and making purchases.
As a simplification, there are no explicit products to track. Only the amounts of each transaction or offer are recorded.
There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.
The basic task is to use the data to identify which groups of people are most responsive to each type of offer, and how best to present each type of offer.

## Data Dictionary

### profile.json
Rewards program users (17000 users x 5 fields)

gender: (categorical) M, F, O, or null
age: (numeric) missing value encoded as 118
id: (string/hash)
became_member_on: (date) format YYYYMMDD
income: (numeric)

### portfolio.json
Offers sent during 30-day test period (10 offers x 6 fields)

reward: (numeric) money awarded for the amount spent
channels: (list) web, email, mobile, social
difficulty: (numeric) money required to be spent to receive reward
duration: (numeric) time for offer to be open, in days
offer_type: (string) bogo, discount, informational
id: (string/hash)

### transcript.json
Event log (306648 events x 4 fields)

person: (string/hash)
event: (string) offer received, offer viewed, transaction, offer completed
value: (dictionary) different values depending on event type
offer id: (string/hash) not associated with any "transaction"
amount: (numeric) money spent in "transaction"
reward: (numeric) money gained from "offer completed"
time: (numeric) hours after start of test

### features.pkl
Saved list from each user`s profile with 'age','gender','income','became_member_on' info that are used as input features to the classification algorithm.

### labels.pkl
Saved list from each user with the indicated offer, based on the transcripts data, that are used as output labels to the classification algorithm.

### [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9].pkl
There are 10 pkl files each one with the model that predict if a given user features is indicated or not to that offer, according to the list:

y[i]     offer_type :                    id  

y[0]           bogo : ae264e3637204a6fb9bb56bc8210ddfd   
y[1]           bogo : 4d5c57ea9a6940dd891ad53e9dbe8da0   
y[2]           bogo : 9b98b8c7a33c4b65b9aebfe6a799e6d9   
y[3]           bogo : f19421c1d4aa40978ebb69ca19b0e20d   
y[4]       discount : 0b1e1539f2cc45b7b9fa7c272da2e1d7   
y[5]       discount : 2298d6c36e964ae4a3e7e9706d1fb8c2   
y[6]       discount : fafdcd668e3743c1bb461111dcafc2a4   
y[7]       discount : 2906b810c7d4411798c6938adc9daaa5   
y[8]  informational : 3f207df678b143eea3cee63160fa8bed   
y[9]  informational : 5a8bc65990b245e5a138643cd4eb9837 

### process_data.ipynb
Jupyter notebook used to process the data, with an exploratory analysis initially, next the data is filtered, a function to evaluate the user log of transcripts is implemented, and the features and labels pkl files are saved.

### model.ipynb
Jupyter notebook used to implement the classification algorithm that predicts whether or not someone will respond to an offer. 10 different models are computed, each one resulting in a separated pkl file related to each offer prediction.

### view_results.ipynb
Jupyter notebook used to check the predicted results. There are one first section which takes the full list of users and compare the predicted results with the labels.pkl file, computing the accuracy, precision and recall for each offer. Next, it is possible to input the features of one user and predict which offer is the best indicated to that profile.

## How to work with the files:
First, run the process_data.ipynb, that takes the portfolio.json, transcript.json and features.pkl and generate the features.pkl and labels.pkl.
Second, run the model.ipynb that takes the features.pkl and labels.pkl and generate the [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9].pkl.
Finally, run the view_results.ipynb, that takes the [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9].pkl and make the predictions.

