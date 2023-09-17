title: Trust Not: Pitfalls in returning model probabilities as confidence scores to end-users in multiclass classification
author: Ronan Diver

# Err.. What?
* picture the scene - rapidly innovating, decide to return probability as a confidence. no one questions it, seems sensible
* soon clients ask what this is and how to use it - it's probability you reply. they tell you that they need it to be 90% for seemingly no reason.
* you start tracking it as an OKR. somehow, when you deploy models with demonstrably higher performance numbers (f1-score etc) the average probability sometimes goes down!
* the okr sits far off what the client tells you they need and it is a frustrating exercise overall
* you write some custom business logic that punishes probabilities when the top two are too close. much later on realise you are probably just a human sigmoid function.

# what is probability? what does it mean in machine learning inference?

* sklearn calibration curves
* easy for binary classification
* worked example of multi class classification
* hierarchichal behaviour can make things harder - re-distribute disallowed label combos. depends on label definition
* difference between the two numbers becomes more important
* clients don't know how to use it to make a decision
* what does 61% vs 62% mean - granularity matters. essentially it turned out we are enabling a binary decision - do i need to review this or not.

# takeaways
* sometimes giving too much information is not useful
* know your audience - if the end user is a data scientist then returning top 3 raw probabilities is probs useful, else not
* 
