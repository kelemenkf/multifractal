# Multifractal

Python library for creating multifractal measures and analysing them. 

Plotting and animating multinomial measures. Beyond 10 iterations matplotlib does not display the measures correctly, but it can be checked that the extreme values of the measure are at the leftmost and rightmost inerval, in the case of the binomial measure. 

Multifractal

attributes:

M - str or list, if list it is the multipliers of a multinomial measure, if a str a type of random multiplier. 

r_type - determines the type of mass conservation of a random multiplier. Options: 'canon' - makes a canonical measure where mass is preserved on average. 'conserv' - makes a conservative measure where mass is conserved exactly at every stage of the construction. If the random multplier is not multinomial it must be set to 'canon'.

loc - location parameter of a random canonical multiplier

scale - scale parameter of a random canonical multiplier

methods: 

Method of moments 

partition_plot(renorm=False)

renorm - Boolean, determines if the plot of the partition function wil be renormalized by vertical replacement to 0. 

-estimate a multinomial measures specturm with iter - integer, will determine the number of iterations 


