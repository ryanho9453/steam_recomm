# Steam video games recommendation

A video games recommendation model using LDA. The dataset comes from kaggle.
Steam Video Games -- (https://www.kaggle.com/tamber/steam-video-games)


# About the Data
Steam is a digital distribution platform for purchasing and playing video games. The dataset contains user's purchasing record, and the challenge is to predict which video game will the user purchase.

The dataset contains the interaction between 12,393 users and 5,155 games. In comparison, 7,672 games were released on steam in 2017.
    
    
# model
We convert the steam purchase record to a user-game matrix, and feed the data to LDA model to make recommendation.

<a href="https://ibb.co/X2SbZBD"><img src="https://i.ibb.co/ZLWV6sS/2019-01-03-4-24-21.png" alt="2019-01-03-4-24-21" border="0"></a>

For evaluation, we pick one purchased game from each user to be the answer, if the recommendation hit the answer, we call it a hit and calculate the hit rate.
We refer to the evaluation method in 
[P Cremonesi, Y Koren (2010)](https://www.researchgate.net/profile/Paolo_Cremonesi/publication/221141030_Performance_of_recommender_algorithms_on_top-N_recommendation_tasks/links/55ef4ac808ae0af8ee1b1bd0.pdf)



# Demo

####  train model

```sh
$ python3 main.py 
```
Hit Rate
<a href="https://ibb.co/bQyGHrM"><img src="https://i.ibb.co/HBM9Nrf/Figure-1.png" alt="Figure-1" border="0"></a>

( Hit rate = recall ) 

Hit example
<a href="https://ibb.co/94DtmdJ"><img src="https://i.ibb.co/PWLQsVR/2019-01-09-4-55-07.png" alt="2019-01-09-4-55-07" border="0"></a>

Miss example
<a href="https://ibb.co/dcHmcSh"><img src="https://i.ibb.co/GJrHJXh/2019-01-09-4-55-35.png" alt="2019-01-09-4-55-35" border="0"></a>
