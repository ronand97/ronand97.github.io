Trust Not: Pitfalls in returning model probabilities as confidence scores to end-users in multiclass classification
======

* TOC
{:toc}

# Err.. What?

Picture the scene. You've just started a project to convert a rules-based prediction system to machine learning. The old system had some sort of number, bound between 0 and 1, representing how good any one prediction was. This number was previously only exposed internally. You are in the rapid innovation phase and settle on using a Random Forest from sklearn. In an act of extemporary genius you decide to return the probabability from the `predict_proba` method of the scikit learn model as a proxy for the _confidence_ of the prediction, which can be returned to the end consumers of the model predictions. No one really questions this at the time - infrastructure and prediction pipelines are built and everyone is happy. 

**The end.**

Ah wait.. A few months in you get a big client coming in asking what the confidence number is and how to use it. Upon promptly informing them that it is the probability of prediction, the client replies by setting an arbritrary probability threshold for which they want (need?) all predictions to be above. You take a quick look at the distribution of your confidences and start to realise they are lower than the threshold, and maybe your multiple classes are not so easily separable by the random forest..

You start tracking the mean average confidence as a team OKR. Somehow when you deploy models with a demonstrably higher performance metric (f1-scores, etc) the average probability sometimes goes down! You end up writing some custom business logic to modify the probabilities to better reflect what the clients need for their processes, and realise you have probably written some sort of sigmoid-esque function. 

If any of this resonates with you, fear not! I have gone through all the pitfalls, and will incriminate myself by discussing them below in the hopes it helps others.


# What is probability? What does it mean in machine learning inference?
_We can all google and use chatGPT, but it would be remiss of me not to include a quick and unrigorous definition._
Probability is a measure of how likely something is to happen. As opposed to deterministic processes (where if you land on the "answer" then it is definitely true), probabilistic processes work on some sort of likelihood of occurrence. 
In classification tasks, the prediction probability represents the likelihood of the incoming data point belonging to the predicted class.

![image](https://www.explainxkcd.com/wiki/images/2/2e/prediction.png)

# Pitfall number 1 - sklearn calibration curves
It turns out you can't fully trust the raw number returned from the `predict_proba_` methods from all models. Pausing to think about this for a few minutes intuitively this makes sense for random forest. RF is an ensemble of multiple decision trees and the probability is the average from all the trees. Although the data seen by each tree is determined by bagging (bootstrap aggregation - involves sampling with replacement which aims to reduce variance) there is still some underlying variation seen by each tree so it becomes very unlikely that all the trees will predict 0 or 1. In fact sklearn has a whole page on this showing how you the effects manifest in practise (RF looks sigmoid-y) and how to perform the calibration. It's an interesting read, and I'll attach the pretty picture below: https://scikit-learn.org/stable/modules/calibration.html.

![image](https://github.com/ronand97/ramblings/assets/45203963/d6119b19-60c4-482f-ac70-010b4c4b3d06)

## Takeaway
Depending on model choice, the default probability values may not be fully accurate. However, ask yourself what level of granularity do you need? Which brings me onto the next pitfall..

# Pitfall number 2 - Granularity
Put yourself in the end-user's shoes - what is the difference between 61% and 62% confidence? How can they use that to make a decision? Would they ever set a threshold to the individual number - e.g. I will trust and use predictions above 64% but not 63% and below? The key here is to **put yourself in your users' shoes** and think about how they will use it. In my project, clients were automating financial decisions (typically underwriting), and when it boiled down to it they just needed a binary decision about whether to use the predictions or whether to manually review them. Implementing this essentially removes some of the decision complexity away for the end-user as we could help them decide whether our predictions were good enough (e.g. inspect the confidence distributions and do some manual spot-checking to assess a rough threshold).

As a side note I had a look around at other companies in my industry (fintech) making predictions and how they handled this (if returning a confidence score). For example, Bud Financial return a decimal confidence score in their API, shown in their example API call, but don't reference it in the documentation at all[^1]. 
Interestingly, Plaid recently announced some new bank transaction enrichments and they have started to return a confidence measure. However, they have decided to bucket the predictions into: very high; high; medium; low; unknown[^2]. This makes more sense to me intuitively, but I would still question the need for 5 levels of granularity. What does it mean for me if the confidence of a prediction is low vs medium?

## Takeaway
Think carefully about who is going to use your predictions. If it is another data professional, maybe returning raw probabilities makes sense. However, if it is a non-technical audience, then simply giving them extra "bonus" information can muddy the waters.

![image](https://imgs.xkcd.com/comics/tmi.png)

[^1]: https://us-api-docs.thisisbud.com/#tag/Enriched-Transactions/operation/v2_transactions_get
[^2]: https://plaid.com/docs/api/products/transactions/#transactions-sync-response-added-personal-finance-category-confidence-level
# Pitfall number 3 - MultiClass Classification makes things harder
Consider we toss a coin 10000 times and record the results and train a model to predict the results (let’s assume the coin can only land on heads or tails). Our probabilities would look like:

| Class | Probability |
|-------|-------------|
| Heads | 50%         |
| Tails | 50%         |

Now consider that we make the coin really thick so there is a chance it lands on its edge, we might see something like:

| Class | Probability |
|-------|-------------|
| Heads | 47.5%       |
| Tails | 47.5%       |
| Edge  | 5%          |

and finally consider we have let some rogue engineer loose on our coin and it now has many edges upon which to land:

| Class | Probability |
|-------|-------------|
| Heads | 40%         |
| Tails | 40%         |
| Edge1 | 5%          |
| Edge2 | 5%          |
| Edge3 | 5%          |
| Edge4 | 5%          |

As we add more possibilities the probability pool gets diluted. The same thinking can be extended to a multi-class classification machine learning model. In an ideal world, each predictable class is fully separable from the next, and any incoming data point will have a prediction made that is 100% probability belonging to one class. In reality, data is usually not so separable and other classes often hold some finite probability. Consider you have a multiclass classification model that can predict 50 classes. The output of `predict_proba` could look like:
| Class   | Probability |
|---------|-------------|
| Class1  | 51%         |
| Class2  | 1%          |
| Class3  | 1%          |
| Class4  | 1%          |
| ...     | ...         |
| Class50 | 1%          |

It is clear that Class1 is the winning prediction by a long shot. However if you return the probability, it doesn't look that great. 51% out of a potential 100%? That might not meet your client's needs, or your internal OKRs. **The difference in probability values between the chosen prediction and the second highest is equally important**. For example, in the example above, if Class1 had 51% probability but Class 2 had 49% (with the rest being 0) that would be a much less confident prediction than the scenario shown above but the given probability from Class 1 would remain the same at 51%.

## Takeaways
* As you add more predictable classes to your models, don't be surprised if the average probability decreases unless the data is very high quality.
* The difference in probability values between the top 2 or 3 highest probability classes may be important to consider (although I haven't personally done it, it is possible you could implement something like this into a loss function so the model tries to optimise for highest probability predictions).

# Pitfall number 4 - Hierarchichal/domain-specific behaviour 
Consider you have a hierarchical structure to your labelled data. For example, a generic classifier that tries to predict 3 levels:
1. Animal, human or inanimate object
2. Type of animal/human/object
3. Subtype of animal/human/object

Example labels for this data could include:
* Animal / Dog / Labrador
* Human / European / Child

The way you design your labels could make the probabilities harder to interpret. You could concatenate them all together (`"Animal_Dog_Labrador"`) and train one model to predict all levels, but that seems a bit silly. It makes more sense to have a model per-level. However, this can cause issues. Consider the following predict_proba outputs for L1 and L2 (the line of thinking is easily extended to L3):

| L1     | Probability |
|--------|-------------|
| Human | 95%         |
| Animal  | 4%          |
| Object | 1%          |


| L2       | Probability |
|----------|-------------|
| English      | 80%         |
| Scottish      | 10%         |
| Cat | 2%          |
| Dog  | 8%          |
| ..       | ..          ||


We see that the obvious prediction from L1 is `Human`. However the L2 probability for `English` is slightly less at 80%, because partially it couldn't decide whether or not it was also potentially a `Cat`, `Scottish` or `Dog` (no insulting implications intended, I just chose a crap example...). We can just return 80% right? Well you could, but we know that it's not possible for `L1==Human` and `L2==Dog`. So we are actually underselling the probability/confidence of our L2 prediction. And what if our models went haywire and predicted a wrong combination? We can use our knowledge of the "allowed" category combinations across L1-L3 to restrict both the predictable categories and proportionally re-distribute the magnitude of the probability between the other allowed categories.

In the above example for L2, only `Scottish` and `English` are visibly allowed. Hence we could re-distribute the 8% probability from `Dog` proportionally between the allowed categories. 

## Takeaways
Hierarchical behaviour can make probabilities and confidences very confusing if all combinations of label aren't allowed. This may have to be accounted for, particularly if you are "underselling" how good your predictions are due to the dilution effects of non-allowed category combinations.

# Concluding points
* Just returning some raw decimal number representing prediction confidence is probably a mistake without taking time to understand how someone would actually use the information (this can probably be extended to any system design).
* Giving users extra bonus information can detract from the prediction
* The link between model performance on a hold-out set and production prediction probability distributions can be tenuous and confusing
* There are a number of "gotchas" associated with packaging probability into a usable confidence score - possibly the most important thing to consider is the level of granularity required.
