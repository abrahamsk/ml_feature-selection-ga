# ml_neural-networks

Experiments in using feature subset selection to improve neural net performance.  

#### Running:
Execute `experiment1_ga.py` from your IDE or command line of choice.  
Key files for running GA-modified neural net:
```
neural-net-ga/
    genetic_algorithm.py  
    neural_net_ga.py  
    experiment1_ga.py  
    experiment1_non_ga.py  
    input.py  
    letter.py  
    letter-recognition.data  
```  

If you're curious, `experiment2.py` and `experiment3.py` follow a similar pattern to exp1 and can be modified in a similar fashion.  


Buyer beware: there's a bug somewhere that comes up when getting test accuracy.  
Also seems to affect accuracy across training for a high number of epochs.  
Setting learning rate `eta`, momentum `alpha` and/or number of hidden units `n` to higher numbers improves accuracy as a stop-gap measure.  
I suspect `back_propagation(...)` is the culprit (I'm looking at you, momentum/delta calculation).  

#### Dependencies
All files mentioned in the `from/import/include ...` statements, especially:  
deap, pyplot, numpy (and scikit is always fun)

#### References
This project was motivated by the work of Yang and Honavar (1997), and Mitchell (*Complexity: A Guided Tour*, 2009).  
Yang and Honavar's paper (*Feature Subset Selection Using a Genetic Algorithm*) is publicly available at http://lib.dr.iastate.edu/cs_techreports/156/  
It has been included in this repo for ease of access to the curious.  Mitchell's work is available at a number of booksellers.  
