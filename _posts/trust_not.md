title: Trust Not: Pitfalls in returning model probabilities as confidence scores to end-users in multiclass classification
author: Ronan Diver

# Err.. What?

Picture the scene. You've just started a project to convert a rules-based prediction system to machine learning. The old system had some sort of number, bound between 0 and 1, representing how good any one prediction was. You are in the rapid innovation phase and settle on using a Random Forest from sklearn. In an act of extemporary genius you decide to return the probabability from the `predict_proba` method of the scikit learn model as a proxy for the _confidence_ of the prediction, which can be returned to the end consumers of the model predictions. No one really questions this at the time - infrastructure and prediction pipelines are built and everyone is happy. 

**The end.**

Ah wait.. A few months in you get a big client coming in asking what the confidence number is and how to use it. Upon promptly informing them that it is the probability of prediction, the client replies by setting an arbritrary probability threshold for which they want (need?) all predictions to be above. You take a quick look at the distribution of your confidences and start to realise they are lower than the threshold, and maybe your multiple classes are not so easily separable by the random forest..

You start tracking the mean average confidence as a team OKR. Somehow when you deploy models with a demonstrably higher performance metric (f1-scores, etc) the average probability sometimes goes down! You end up writing some custom business logic to modify the probabilities to better reflect what the clients need for their processes, and realise you have probably some sort of sigmoid function. 

If any of this resonates with you, fear not! I have gone through all the pitfalls, and will incriminate myself by discussing them below in the hopes it helps others.


# what is probability? what does it mean in machine learning inference?
I know we can all google and use chat GPT but it would be remiss not to include a quick , unrigorous definition. Probability is a measure of how likely something is to happen. As opposed to deterministic processes (where if you land on the "answer" then it is definitely true), probabilistic processes work on some sort of likelihood of occurrence. 
In classification tasks, the prediction probability represents the likelihood of the incoming data point belonging to the predicted class.

# Pitfall number 1 - sklearn calibration curves
It turns out you can't fully trust the raw number returned from the `predict_proba_` methods from all models. Pausing to think about this for a few minutes intuitively this makes sense for random forest. RF is an ensemble of multiple decision trees and the probability is the average from all the trees. Although the data seen by each tree is determined by bagging (bootstrap aggregation - involves sampling with replacement) there is still some underlying variation seen by each tree so it becomes very unlikely that all the trees will predict 0 or 1. In fact sklearn has a whole page on this showing how you the effects manifest in practise (RF looks sigmoid-y) and how to perform the calibration. It's an interesting read, and I'll attach the pretty picture below: https://scikit-learn.org/stable/modules/calibration.html.

![image](https://github.com/ronand97/ramblings/assets/45203963/d6119b19-60c4-482f-ac70-010b4c4b3d06)

Takeaway: depending on model choice, the default probability values may not be fully accurate. However, ask yourself what level of granularity do you need? Which brings me onto the next pitfall..

# Pitfall number 2 - Granularity

* clients don't know how to use it to make a decision
* what does 61% vs 62% mean - granularity matters. essentially it turned out we are enabling a binary decision - do i need to review this or not.

# Pitfall number 3 - MultiClass Classification makes things harder
* easy for binary classification
* worked example of multi class classification
* difference between the two numbers becomes more important

  
# Pitfall number 4 - Hierarchichal/domain-specific behaviour 
* hierarchichal behaviour can make things harder - re-distribute disallowed label combos. depends on label definition


# takeaways
* sometimes giving too much information is not useful
* know your audience - if the end user is a data scientist then returning top 3 raw probabilities is probs useful, else not
* 
