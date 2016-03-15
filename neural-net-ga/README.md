# ml_neural-networks

Experiments in using feature subset selection to improve neural net performance.  
See: *Feature Subset Selection Using a Genetic Algorithm* (Yang, Honavar 1997)

#### Running:
Execute `experiment1_ga.py` from your IDE or command line of choice.  
Key files for running GA-modified neural net:
```
neural-net-ga/
    genetic_algorithm.py  
    neural_net_ga.py  
    experiment1_ga.py
```  

If you're curious, `experiment2.py` and `experiment3.py` follow a similar pattern to exp1 and can be modified in a similar fashion.  


Buyer beware: there's a bug somewhere that comes up when getting test accuracy.  
Also seems to affect accuracy across training for a high number of epochs.  
Setting learning rate `eta`, momentum `alpha` and/or number of hidden units `n` to higher numbers improves accuracy as a stop-gap measure.  
I suspect `back_propagation(...)` is the culprit (I'm looking at you, momentum/delta calculation).  

#### Dependencies
All files mentioned in the `from/import/include ...` statements, especially:  
deap, pyplot, numpy (and scikit is always fun)