# Steam video games recommendation

The model recommend video games to steam user. The dataset comes from kaggle.
Steam Video Games -- (https://www.kaggle.com/tamber/steam-video-games)


<a href="https://ibb.co/6YqgCQ1"><img src="https://i.ibb.co/F6dgTp0/2019-01-03-12-13-45.png" alt="2019-01-03-12-13-45" border="0"></a>

# Overview

  - Input some keyword and the model will recommend some relevant keyword
  - The model is trained with chinese news data, so it's applicable to search engine of  online news website


You can also:
  - Customize your model by tuning the parameter in config.json or use your own news data
  - Implant prior knowledge to indirectly affect the learning process and customize the recommendation results

# About the Data
Steam is a digital distribution platform for purchasing and playing video games.
The dataset contains user's purchasing record, and the challenge is to predict 
which video game will the user purchase.

The dataset contains the interaction between 12,393 users and 5,155 games.
In comparison, 7,672 games were released on steam in 2017.
    
    
# model
We convert the steam purchase record to a user-game matrix, and feed the data to LDA model
to make recommedation.

imgae

For evaluation, we pick one purchased game from each user to be the answer,
If the recommendation hit the answer, we call it a hit and calculate the hit rate.
We refer to the evaluation method in 
[P Cremonesi, Y Koren (2010)](https://www.researchgate.net/profile/Paolo_Cremonesi/publication/221141030_Performance_of_recommender_algorithms_on_top-N_recommendation_tasks/links/55ef4ac808ae0af8ee1b1bd0.pdf)



# Demo


```sh
--mode train(train a new model) / recommend (recommend with saved model)
--model LDA / RNN (choose a model RNN or LDA)
--word (input your word in recommend mode)
```
               
#### 1. ask for a recommendation with saved model


```sh
$ python3 main.py --mode recommend --model LDA --word (input your word here)
```

For production environments...

#### 2. train a new model

```sh
$ python3 main.py --mode train --model LDA 
```

