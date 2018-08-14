##################################################################################################

## File name: multistepnet_sim.R
## Programmer: Elizabeth Chase
## Project: Multi-step elastic net, in collaboration with Phil Boonstra
## Date: Worked on from Oct. 1, 2017-June 1, 2018; this polished code was completed and assembled
##       for publication on May 17-May 18, 2018
## Other related files: multistepnet_functions.R, multistepnet_example.R
## Purpose: This file simulates data according to 12 scenarios and then performs an elastic net,
##          underpenalized elastic net, multi-step elastic net, IPF-Lasso, and sparse group lasso
##          on each simulated dataset. 

#################################################################################################

#Loading packages
library(parallel);
library(pROC);
library(stringr);
library(glmnet);
library(SGL);

#Set working directory:
setwd("~/Desktop/Research/Phil Elastic Net/Final code");

#Set source for multistepnet_functions_revised:
source("~/Desktop/Research/Phil Elastic Net/Final code/multistepnet_functions_revised.R");

# Calculate the number of cores
no_cores <- detectCores() - 1

# Initiate cluster
cl <- makeCluster(no_cores)

#The desired number of replicates goes here:
reps <- 500

#The true intercept for our model goes here:
trueintercept <- -1.39 #corresponds to prevalence of 0.2 for the average person

#We pick a sample size of 200
n <- 200

#Now we declare our covariate values--these are different for each scenario. Depending on which scenario
#you want, remove the comment symbols in front of the statements. Truebeta_known are the established covariates,
#while truebeta_myst are the unestablished covariates.

#Scenario 1A
truebeta_known <- rep(0.26,10)
truebeta_myst <- rep(0,30)

#Scenario 1B
#truebeta_known <- rep(0.2,10)
#truebeta_myst <- c(0.6, rep(0,29))

#Scenario 1C
#truebeta_known <- rep(0.25,10)
#truebeta_myst <- c(rep(0.05, 5), rep(0,25))

#Scenario 2A
#truebeta_known <- rep(0.26, 10)
#truebeta_myst <- rep(0,90)

#Scenario 2B
#truebeta_known <- rep(0.2, 10)
#truebeta_myst <- c(0.6, rep(0,89))

#Scenario 2C
#truebeta_known <- rep(0.25, 10)
#truebeta_myst <- c(rep(0.05, 5), rep(0,85))

#Scenario 3A
#truebeta_known <- rep(0.14, 20)
#truebeta_myst <- rep(0,480)

#Scenario 3B
#truebeta_known <- rep(0.11, 20)
#truebeta_myst <- c(0.6, rep(0,479))

#Scenario 3C
#truebeta_known <- rep(0.13, 20)
#truebeta_myst <- c(rep(0.05,5),rep(0,475))

#Scenario 4A
#truebeta_known <- c(rep(0.26, 10),rep(0,10))
#truebeta_myst <- rep(0,480)

#Scenario 4B
#truebeta_known <- c(rep(0.2,10),rep(0,10))
#truebeta_myst <- c(0.6, rep(0,479))

#Scenario 4C
#truebeta_known <- c(rep(0.25, 10),rep(0,10))
#truebeta_myst <- c(rep(0.05, 5), rep(0,475))

p1 = length(truebeta_known); 
which_set1 = 1:p1; 
p2 = length(truebeta_myst); 
which_set2 = p1 + (1:p2); 
truebetas = c(truebeta_known,truebeta_myst); #concatenate 

#And for future use, we will find the indices of the true/false and unknown/known covariates:
trueind <- which(truebetas != 0)
trueeind <- which(truebetas[which_set1] != 0)
trueuind <- which(truebetas[which_set2] != 0) + p1

falseind <- which(truebetas == 0)
falseestind <- unique(c(which_set1, which(truebetas==0)))
falseunind <- unique(c(which_set2, which(truebetas==0)))

#We put the compound symmetric correlation here:
pairwise_correlation = 0.2;

#Now we will simulate the design matrices:
myx <- as.list(rep(1,reps))
all_x <- mclapply(myx, makex, mc.preschedule = TRUE, mc.set.seed = TRUE,
                  mc.silent = TRUE, mc.cores = no_cores, mc.cleanup = TRUE)

#Now we will simulate the outcome vector and append it to the design matrix:
bigdesign <- mclapply(all_x, makey, mc.preschedule = TRUE, mc.set.seed = TRUE,
                      mc.silent = TRUE, mc.cores = no_cores, mc.cleanup = TRUE)

#And finally, we will simulate a test dataset for AUC calculations:
testsetx <- matrix(rnorm(n * (p1 + p2)), nrow = n)%*%chol(diag(1 - pairwise_correlation,(p1+p2)) + pairwise_correlation)
testsety <- rbinom(n, 1, expit(trueintercept + testsetx%*%truebetas))

#We set penalized regression parameters: our alpha sequence, number of CV replicates, and number of folds:
alpha_seq = c(0.1, 0.5, 0.9)
n_cv_rep = 25;
n_folds = 5;

#Now running the multi-step net:
mynet <- mclapply(bigdesign,donet,mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                  mc.cores=no_cores,mc.cleanup=TRUE)

#And the SGL:
mysgl <- mclapply(bigdesign,dosgl,mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                  mc.cores=no_cores,mc.cleanup=TRUE)

#We would want to save this workspace for each scenario
#save.image('/home/ecchase/ElasticNet/Scen.RData')

#To process the results from this code, we would move to multistepnet_eval.R
