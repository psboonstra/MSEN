# MSEN

### Current Suggested Citation

In progress

DOI for this repository:

In progress

## Executive Summary

This repository contains code for conducting the simulation study in Chase and Boonstra (2019). 

## Further details

In more detail, there are three <samp>R</samp> scripts (ending in  <samp>.R</samp>) included in this repository (in addition to this README). 


<samp>functions.R</samp> provides all of the necessary functions to fit the (i) multistep elastic net via the function <samp>donet</samp>, (ii) the IPF Lasso (i.e. a multistep elastic net with tuning parameter &lambda; set to 1) also via the function <samp>donet</samp>, and (iii) the (sparse) group lasso via <samp>sgl</samp>. 


<samp>run_sims.R</samp> conducts a simulation of a given scenario. 

<samp>eval_sims.R</samp> gives the code to create the figures and tables in the manuscript and supplementary material reporting on the simulation study. 

