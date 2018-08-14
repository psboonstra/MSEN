###################################################################################################

## File name: multipstepnet_functions_revised.R
## Programmer: Elizabeth Chase
## Project: Multi-step elastic net, in collaboration with Phil Boonstra
## Date: Worked on from Oct. 1, 2017-June 1, 2018; this polished code was completed and assembled
##       for publication on May 17-May 18, 2018
## Revisions from multistepnet_functions.R occurred on June 6, 2018
## Other related files: multistepnet_sim.R, multistepnet_eval.R, MSexample_functions.R
## Purpose: This file contains R functions that perform an elastic net, underpenalized elastic net, 
##          multi-step elastic net, IPF-Lasso, and sparse group lasso. It also contains functions
##          to process the output from the lasso functions and assess AUC, Brier score, sensitivity,
##          and TDR for outputted data. 

###################################################################################################

## PART 1: FUNCTIONS TO SIMULATE DATA

#expit is needed for the logistic regression setting 
expit = function(x) {1/(1+exp(-x));}

#makex simulates a design matrix according to our preset coefficient, sample size, and correlation values:
makex <- function(x) {x <- x*matrix(rnorm(n * (p1 + p2)), nrow = n)%*%chol(diag(1 - pairwise_correlation,(p1+p2)) + pairwise_correlation)
}

#makey creates an outcome matrix based on the design matrices simulated above, and appends it to the design
#matrix:
makey <- function(design) {xy <- cbind(design, rbinom(n, 1, expit(trueintercept + design%*%truebetas)));}

## PART 2: FUNCTIONS TO PERFORM PENALIZED REGRESSION ON SIMULATED DATA

#donet performs all 5 versions of the elastic net. It fits the following 5 models:
    #Model 1: phi_1 = 0, phi_2 = 1
    #Model 2: phi_1 = 1/16, phi_2 = 1
    #Model 3: phi_1 = 1/2, phi_2 = 1
    #Model 4: phi_1 = 1, phi_2 = 1
    #Model 5: phi_1 = 1, phi_2 = 0
#And then produces the final models:
  #Elastic net: Model 4
  #IPF-EN: best of models 2, 3, and 4
  #IPF-EN + Zero: best of models 1, 2, 3, and 4
  #IPF-EN + Infinity: best of models 2, 3, 4, and 5
  #MSN: best of models 1, 2, 3, 4, and 5

#the input is a design matrix and other variables used in multistepnet_sim.R

donet <- function(x){
  #We separate out the design matrix and outcome vector:
  all_x <- x[,1:(p1+p2)]
  y <- x[,(p1+p2+1)]
  
  #First, we initialize values to hold the deviance and lambda sequence for each
  #combination of alpha and phi (3 alphas and 5 phis)
  for(j in 1:length(alpha_seq)) {#initialize values
    assign(paste0("alpha",j,"_model1_dev"),0);
    assign(paste0("alpha",j,"_model2_dev"),0);
    assign(paste0("alpha",j,"_model3_dev"),0);
    assign(paste0("alpha",j,"_model4_dev"),0);
    assign(paste0("alpha",j,"_model5_dev"),0);
    assign(paste0("alpha",j,"_model1_lambda_seq"),NULL);
    assign(paste0("alpha",j,"_model2_lambda_seq"),NULL);
    assign(paste0("alpha",j,"_model3_lambda_seq"),NULL);
    assign(paste0("alpha",j,"_model4_lambda_seq"),NULL);
    assign(paste0("alpha",j,"_model5_lambda_seq"),NULL);
  }
  
  #We standardize our design matrix:
  center_all_x = apply(all_x,2,mean,na.rm=T);
  scale_all_x = apply(all_x,2,sd,na.rm=T);
  std_all_x = scale(all_x,center = center_all_x, scale = scale_all_x);
  
  #Because glmnet allows us to assign folds, we assign observations to folds to ensure
  #consistency across imputations. 
  foldid = matrix(NA,length(y),n_cv_rep);
  for(i in 1:n_cv_rep) {
    foldid[,i] = sample(rep(1:n_folds,length = length(y)));
  }
  
  #We now fit our 5 differently-penalized models for each of our 11 values of alpha for each of our 
  #cross-validated replicates:
  
  for(i in 1:n_cv_rep) {
    for(j in 1:length(alpha_seq)) {
  
      #Model 1: phi_1 = 0, phi_2 = 1 (no penalization on established and full on unestablished)
      penalize1 <- rep(1,(p1+p2))
      penalize1[1:p1] <- 0
      penalize1[(p1+1):(p1+p2)] <- 1
      assign(paste0("fitted1_mod",j),
             cv.glmnet(x = std_all_x, 
                       y = factor(1*y),
                       standardize = F,
                       family = "binomial",
                       alpha = alpha_seq[j],
                       foldid = foldid[,i],
                       lambda = get(paste0("alpha",j,"_model1_lambda_seq")),
                       penalty.factor = penalize1,
                       keep = T));
      assign(paste0("alpha",j,"_model1_dev"),get(paste0("alpha",j,"_model1_dev")) + get(paste0("fitted1_mod",j))$cvm/n_cv_rep);
      if(is.null(get(paste0("alpha",j,"_model1_lambda_seq")))) {assign(paste0("alpha",j,"_model1_lambda_seq"),get(paste0("fitted1_mod",j))$lambda);}
      fitted_prob = get(paste0("fitted1_mod",j))$fit.preval[,1:length(get(paste0("alpha",j,"_model1_lambda_seq")))];
  
      #Model 2 (phi_1 = 1/16, phi_2 = 1, or small penalization on the established covariates)
      penalize2 <- rep(1,(p1+p2))
      penalize2[1:p1] <- (1/16)
      penalize2[(p1+1):(p1+p2)] <- 1
      assign(paste0("fitted2_mod",j),
             cv.glmnet(x = std_all_x, 
                       y = factor(1*y),
                       standardize = F,
                       family = "binomial",
                       alpha = alpha_seq[j],
                       foldid = foldid[,i],
                       lambda = get(paste0("alpha",j,"_model2_lambda_seq")),
                       penalty.factor = penalize2,
                       keep = T));
      assign(paste0("alpha",j,"_model2_dev"),get(paste0("alpha",j,"_model2_dev")) + get(paste0("fitted2_mod",j))$cvm/n_cv_rep);
      if(is.null(get(paste0("alpha",j,"_model2_lambda_seq")))) {assign(paste0("alpha",j,"_model2_lambda_seq"),get(paste0("fitted2_mod",j))$lambda);}
      fitted_prob = get(paste0("fitted2_mod",j))$fit.preval[,1:length(get(paste0("alpha",j,"_model2_lambda_seq")))];
      
      #Model3 (Phi_1 = 1/2, Phi_2 = 1, or double penalization on the unestablished covariates)
      penalize3 <- rep(1,(p1+p2))
      penalize3[1:p1] <- 1/2
      penalize3[(p1+1):(p1+p2)] <- 1
      assign(paste0("fitted3_mod",j),
             cv.glmnet(x = std_all_x, 
                       y = factor(1*y),
                       standardize = F,
                       family = "binomial",
                       alpha = alpha_seq[j],
                       foldid = foldid[,i],
                       lambda = get(paste0("alpha",j,"_model3_lambda_seq")),
                       penalty.factor = penalize3,
                       keep = T));
      assign(paste0("alpha",j,"_model3_dev"),get(paste0("alpha",j,"_model3_dev")) + get(paste0("fitted3_mod",j))$cvm/n_cv_rep);
      if(is.null(get(paste0("alpha",j,"_model3_lambda_seq")))) {assign(paste0("alpha",j,"_model3_lambda_seq"),get(paste0("fitted3_mod",j))$lambda);}
      fitted_prob = get(paste0("fitted3_mod",j))$fit.preval[,1:length(get(paste0("alpha",j,"_model3_lambda_seq")))];
   
      #Model 4: phi_1 = 1, phi_2 = 1, or no penalization on the established covariates
      assign(paste0("fitted4_mod",j),
             cv.glmnet(x = std_all_x, 
                       y = factor(1*y),
                       standardize = F,
                       family = "binomial",
                       alpha = alpha_seq[j],
                       foldid = foldid[,i],
                       lambda = get(paste0("alpha",j,"_model4_lambda_seq")),
                       keep = T));
      assign(paste0("alpha",j,"_model4_dev"),get(paste0("alpha",j,"_model4_dev")) + get(paste0("fitted4_mod",j))$cvm/n_cv_rep);
      if(is.null(get(paste0("alpha",j,"_model4_lambda_seq")))) {assign(paste0("alpha",j,"_model4_lambda_seq"),get(paste0("fitted4_mod",j))$lambda);}
      fitted_prob = get(paste0("fitted4_mod",j))$fit.preval[,1:length(get(paste0("alpha",j,"_model4_lambda_seq")))];
   
      #Model5 (Phi_1 = 1, Phi_2 = 0, or infinite penalization on the unestablished covariates)
      assign(paste0("fitted5_mod",j),
             cv.glmnet(x = std_all_x[,which_set1,drop=F],
                       y = factor(1*y),
                       standardize = F,
                       family = "binomial",
                       alpha = alpha_seq[j],
                       foldid = foldid[,i],
                       lambda = get(paste0("alpha",j,"_model5_lambda_seq")),
                       keep = T));
      assign(paste0("alpha",j,"_model5_dev"),get(paste0("alpha",j,"_model5_dev")) + get(paste0("fitted5_mod",j))$cvm/n_cv_rep);
      if(is.null(get(paste0("alpha",j,"_model5_lambda_seq")))) {assign(paste0("alpha",j,"_model5_lambda_seq"),get(paste0("fitted5_mod",j))$lambda);}
      fitted_prob = get(paste0("fitted5_mod",j))$fit.preval[,1:length(get(paste0("alpha",j,"_model5_lambda_seq")))];
      
    }
    cat(i,"\n");
  }
  
  #We now initialize a 5 x 3 matrix to hold the deviance values for each combination of phi and alpha--15 models total--
  #and then fill it with the deviance values obtained above
  
  best_dev_all = matrix(NA,5,length(alpha_seq), dimnames=list(paste0("Penalty",1:5),paste0("alpha",alpha_seq)));
  for(j in 1:length(alpha_seq)) {
    best_dev_all["Penalty1",j] = get(paste0("alpha",j,"_model1_dev"))[which.min(get(paste0("alpha",j,"_model1_dev")))];
    best_dev_all["Penalty2",j] = get(paste0("alpha",j,"_model2_dev"))[which.min(get(paste0("alpha",j,"_model2_dev")))];
    best_dev_all["Penalty3",j] = get(paste0("alpha",j,"_model3_dev"))[which.min(get(paste0("alpha",j,"_model3_dev")))];
    best_dev_all["Penalty4",j] = get(paste0("alpha",j,"_model4_dev"))[which.min(get(paste0("alpha",j,"_model4_dev")))];
    best_dev_all["Penalty5",j] = get(paste0("alpha",j,"_model5_dev"))[which.min(get(paste0("alpha",j,"_model5_dev")))];
  }
  
  #identify the best (smallest) deviance across all values of alpha for each penalty value; this is the selected alpha
  #for each phi
  which_alpha_all = apply(best_dev_all[c("Penalty1","Penalty2","Penalty3","Penalty4","Penalty5"),],1,which.min);
  #
  #For each penalty, we use the best alpha to obtain that alpha's lambdaseq, and then choose the best lambda for that alpha 
  model1_lambda_seq = get(paste0("alpha",which_alpha_all["Penalty1"],"_model1_lambda_seq"));
  model1_which_lambda = which.min(get(paste0("alpha",which_alpha_all["Penalty1"],"_model1_dev")));
  model1_lambda_select = model1_lambda_seq[model1_which_lambda];
  #
  model2_lambda_seq = get(paste0("alpha",which_alpha_all["Penalty2"],"_model2_lambda_seq"));
  model2_which_lambda = which.min(get(paste0("alpha",which_alpha_all["Penalty2"],"_model2_dev")));
  model2_lambda_select = model2_lambda_seq[model2_which_lambda];
  #
  model3_lambda_seq = get(paste0("alpha",which_alpha_all["Penalty3"],"_model3_lambda_seq"));
  model3_which_lambda = which.min(get(paste0("alpha",which_alpha_all["Penalty3"],"_model3_dev")));
  model3_lambda_select = model3_lambda_seq[model3_which_lambda];
  #
  model4_lambda_seq = get(paste0("alpha",which_alpha_all["Penalty4"],"_model4_lambda_seq"));
  model4_which_lambda = which.min(get(paste0("alpha",which_alpha_all["Penalty4"],"_model4_dev")));
  model4_lambda_select = model4_lambda_seq[model4_which_lambda];
  #
  model5_lambda_seq = get(paste0("alpha",which_alpha_all["Penalty5"],"_model5_lambda_seq"));
  model5_which_lambda = which.min(get(paste0("alpha",which_alpha_all["Penalty5"],"_model5_dev")));
  model5_lambda_select = model5_lambda_seq[model5_which_lambda];
  
  #We standardize our simulated test set:
  std_test = scale(testsetx,center = center_all_x, scale = scale_all_x);
  
  #And now we fit the final models using our optimal values of lambda and alpha for each penalty type:
  model1_fitted_mod = glmnet(x = std_all_x, 
                             y = factor(1*y),
                             standardize = F,
                             family = "binomial",
                             alpha = alpha_seq[which_alpha_all["Penalty1"]],
                             lambda = model1_lambda_seq,
                             penalty.factor = penalize1);

  coef1 = coef(model1_fitted_mod)[,model1_which_lambda];
  mycoef1 <- coef1[-1]/scale_all_x;
  myfit1 <- predict(model1_fitted_mod, std_test, s=model1_lambda_select, type="link")
  
  model2_fitted_mod = glmnet(x = std_all_x, 
                             y = factor(1*y),
                             standardize = F,
                             family = "binomial",
                             alpha = alpha_seq[which_alpha_all["Penalty2"]],
                             lambda = model2_lambda_seq, 
                             penalty.factor = penalize2);
  
  coef2 <- coef(model2_fitted_mod)[,model2_which_lambda];
  mycoef2 <- coef2[-1]/scale_all_x;
  myfit2 <- predict(model2_fitted_mod, std_test, s=model2_lambda_select, type="link")
  
  model3_fitted_mod = glmnet(x = std_all_x, 
                             y = factor(1*y),
                             standardize = F,
                             family = "binomial",
                             alpha = alpha_seq[which_alpha_all["Penalty3"]],
                             lambda = model3_lambda_seq,
                             penalty.factor = penalize3);
           
  coef3 = coef(model3_fitted_mod)[,model3_which_lambda];
  mycoef3 <- coef3[-1]/scale_all_x;
  myfit3 <- predict(model3_fitted_mod, std_test, s=model3_lambda_select, type="link")
  
  model4_fitted_mod = glmnet(x = std_all_x, 
                             y = factor(1*y),
                             standardize = F,
                             family = "binomial",
                             alpha = alpha_seq[which_alpha_all["Penalty4"]],
                             lambda = model4_lambda_seq);
            
  coef4 = coef(model4_fitted_mod)[,model4_which_lambda];
  mycoef4 <- coef4[-1]/scale_all_x;
  myfit4 <- predict(model4_fitted_mod, std_test, s=model4_lambda_select, type="link")
  
  model5_fitted_mod = glmnet(x = std_all_x[,which_set1,drop=F],
                             y = factor(1*y),
                             standardize = F,
                             family = "binomial",
                             alpha = alpha_seq[which_alpha_all["Penalty5"]],
                             lambda = model5_lambda_seq);
        
  coef5 = 0 * coef1;
  coef5[rownames(coef(model5_fitted_mod))] <- coef(model5_fitted_mod)[,model5_which_lambda];
  mycoef5 <- coef5[-1]/scale_all_x;
  myfit5 <- predict(model5_fitted_mod, std_test[,which_set1,drop=F], s=model5_lambda_select, type="link")
 
  #Finally, we determine which of the five models is best overall for each penalty combination:
  myen <- 4
  myipf_en <- which.min(apply(best_dev_all[c("Penalty2","Penalty3","Penalty4"),],1,min));
  myipf_zero <- which.min(apply(best_dev_all[c("Penalty1","Penalty2","Penalty3","Penalty4"),],1,min));
  myipf_inf <- which.min(apply(best_dev_all[c("Penalty2","Penalty3","Penalty4","Penalty5"),],1,min));
  myMS <- which.min(apply(best_dev_all,1,min));
  
  mymodels <- list(mycoef1, myfit1, mycoef2, myfit2, mycoef3, myfit3, mycoef4, myfit4, 
                   mycoef5, myfit5, myen, myipf_en, myipf_zero, myipf_inf, myMS, model1_lambda_select,
                   model2_lambda_select, model3_lambda_select, model4_lambda_select, 
                   model5_lambda_select, alpha_seq[which_alpha_all["Penalty1"]], 
                   alpha_seq[which_alpha_all["Penalty2"]], alpha_seq[which_alpha_all["Penalty3"]],
                   alpha_seq[which_alpha_all["Penalty4"]], alpha_seq[which_alpha_all["Penalty5"]]);
  
  return(mymodels)
}

#dosgl performs a sparse-group lasso on a design matrix and outcome vector

dosgl <- function(x){
  
  #We separate out the design matrix and outcome vector
  all_x <- x[,1:(p1+p2)]
  y <- x[,(p1+p2+1)]
  
  #We initialize values to store the deviance and lambda sequence 
  assign(paste0("modelsgl_dev"),0);
  assign(paste0("modelsgl_lambda_seq"),NULL);
  assign(paste0("modelgroup_dev"),0);
  assign(paste0("modelgroup_lambda_seq"),NULL);
  
  #We standardize our design matrix, make the response vector a factor, and assemble as a list:
  center_all_x = apply(all_x,2,mean,na.rm=T);
  scale_all_x = apply(all_x,2,sd,na.rm=T);
  std_all_x = scale(all_x,center = center_all_x, scale = scale_all_x);
  resp = factor(1*y)
  mydat = list(x=all_x, y=y)
  
  #We create a vector to tell SGL which group each covariate belongs to:
  estab <- rep(1,p1)
  unestab <- rep(2,p2)
  myindex <- c(estab,unestab)
  
  #Then we cross-validate and fit a model for each alpha
  for(i in 1:n_cv_rep) {
      #First fitting a sparse group lasso with alpha = 0.95:
      assign(paste0("fitted_sgl"),
             cvSGL(data=mydat, 
                   index=myindex, 
                   type="logit",
                   standardize = F,
                   alpha = 0.95,
                   nfold = n_folds));
      assign(paste0("modelsgl_dev"),get(paste0("modelsgl_dev")) + get(paste0("fitted_sgl"))$lldiff/n_cv_rep);
      if(is.null(get(paste0("modelsgl_lambda_seq")))) {assign(paste0("modelsgl_lambda_seq"),get(paste0("fitted_sgl"))$fit$lambdas);}
      
      
      #And now fitting a grouped lasso with alpha = 0:
      assign(paste0("fitted_group"),
             cvSGL(data=mydat, 
                   index=myindex, 
                   type="logit",
                   standardize = F,
                   alpha = 0,
                   nfold = n_folds));
      assign(paste0("modelgroup_dev"),get(paste0("modelgroup_dev")) + get(paste0("fitted_group"))$lldiff/n_cv_rep);
      if(is.null(get(paste0("modelgroup_lambda_seq")))) {assign(paste0("modelgroup_lambda_seq"),get(paste0("fitted_group"))$fit$lambdas);}
      
    cat(i,"\n");
  }
  
  #We locate the best lambda value for each model
  sgl_which_lambda = which.min(get(paste0("modelsgl_dev")));
  sgl_lambda_select = modelsgl_lambda_seq[sgl_which_lambda];
  
  group_which_lambda = which.min(get(paste0("modelgroup_dev")));
  group_lambda_select = modelgroup_lambda_seq[group_which_lambda];
  
  #Now we fit the final models, using our best values of alpha and lambda:
  fitted_sgl = SGL(data=mydat,index=myindex, type="logit",
                   standardize = F,
                   alpha = 0.95,
                   lambdas= modelsgl_lambda_seq);
  
  fitted_group = SGL(data=mydat,index=myindex, type="logit",
                   standardize = F,
                   alpha = 0,
                   lambdas= modelgroup_lambda_seq);
  
  #We standardize the test set:
  std_test = scale(testsetx,center = center_all_x, scale = scale_all_x);
  
  #We obtain coefficients and predicted values for the models:
  coef_sgl <- fitted_sgl$beta[,sgl_which_lambda];
  mycoef_sgl <- coef_sgl[-1]/scale_all_x;
  myfit_sgl <- predictSGL(fitted_sgl, testsetx,sgl_which_lambda)
  
  coef_group <- fitted_group$beta[,group_which_lambda];
  mycoef_group <- coef_group[-1]/scale_all_x;
  myfit_group <- predictSGL(fitted_group, testsetx,group_which_lambda)
  
  mymodels <- list(mycoef_sgl, myfit_sgl, sgl_lambda_select, mycoef_group, myfit_group, group_lambda_select)
  
  return(mymodels)
}

## PART 3: FUNCTIONS TO USE OUTPUT FROM PENALIZED REGRESSION FUNCTIONS AND ASSESS PERFORMANCE

#is.integer0 is needed to identify vectors of length 0
is.integer0 <- function(x)
{is.integer(x) && length(x) == 0L}

#subnet is used on the mynet output to extract the different net penalties

####THIS IS WHERE I MADE MY MISTAKE: note that for quelmod=12 and quelmod=14, I'm 
#adding 1 to the penalty selection indicator; that's because, for those methods, it's 
#selecting from a reduced set of penalty options that starts at 2, rather than 1, so
#adding 1 is necessary to get the intended penalty winner. Originally, I wasn't doing 
#that, so if the actual winning penalty was Penalty 2, it was being registered as Penalty 1, 
#or Penalty 4 was being reported as Penalty 3. Adding 1 hopefully fixed the problem.

subnet <- function(x,quelmod){ 
    if (quelmod==11){
      model_pick <- x[[quelmod]]
    } else if (quelmod==12){
      model_pick <- (x[[quelmod]]+1)
    } else if (quelmod==13){
      model_pick <- x[[quelmod]]
    } else if (quelmod==14){
      model_pick <- (x[[quelmod]]+1)
    } else if (quelmod==15){
      model_pick <- x[[quelmod]]
    }
  
  indicator <- model_pick*2
  mymodel <- list(x[[(indicator-1)]], x[[indicator]])
  
  return(mymodel)
}

#subsgl is used on the mysgl output to extract the different sgl penalties
subsgl <- function(x,quelmod)
{ if (quelmod==1){
  mymodel <- list(x[[1]], x[[2]])
} else if (quelmod==2){
  mymodel <- list(x[[4]], x[[5]])
}
  
return(mymodel)
}

#getys obtains the predicted values nets and IPF and then de-logits them
getys <- function(x)
{ predyraw <- x[[2]]
predy <- exp(predyraw)/(1+exp(predyraw))
return(predy)
}

#sglgetys obtains the predicted values for the sparse group lasso from the mysgl output; note
#that the dosgl function already expits the predicted values, so we don't need to do so
sglgetys <- function(x)
{ predy <- x[[2]]
return(predy)
}

#BrierScore uses the predicted values obtained with the getys family of functions above to calculate
#the Brier Score
BrierScore <- function(x)
{bs <- mean((x-testsety)**2)
return(bs)
}

#getAUC uses the predicted values obtained with the getys family of functions above to calculate
#the AUC; note that it uses the pROC package to do so
getAUC <- function(x)
{ myAUC <- roc(response=testsety, predictor=x, smooth=FALSE, auc=TRUE, ci=TRUE, plot=FALSE)
return(myAUC$auc)
}

# covariateeffect calculates the mean estimated effect over the established and unestablished covariates 
covariateeffect <- function(x)
{   small_number = .Machine$double.eps^0.5
    mycoefs <- x[[1]] #extracting coefficients and removing intercept
    if (is.null(names(mycoefs))==0){
      if (names(mycoefs)[1]=="(Intercept)"){
        mycoefs <- mycoefs[2:(p1+p2+1)]
      }
    }  

    esteffect <- mean(mycoefs[which_set1]) #finding the mean signal across the established coefficients
    if (sum(is.na(mycoefs[which_set2]))==p2){unesteffect <- 0 #finding the mean signal across unestablished coefficients
    } else {unesteffect <- mean(mycoefs[which_set2])
    }  
    
    results <- c(esteffect, unesteffect)
    return(results)
}

#findRMSE calculates the root mean squared error
findRMSE <- function(x){
  mycoefs <- x[[1]] #extracting coefficients and removing intercept
  if (is.null(names(mycoefs))==0){
    if (names(mycoefs)[1]=="(Intercept)"){
      mycoefs <- mycoefs[2:(p1+p2+1)]
    }
  }  
  
  estcoefs <- mycoefs[which_set1]
  unestcoefs <- mycoefs[which_set2]
  estRMSE <- sqrt(sum((estcoefs-truebeta_known)^2))
  unestRMSE <- sqrt(sum((unestcoefs-truebeta_myst)^2))
  RMSE <- c(estRMSE, unestRMSE)
  
  return(RMSE)
}

#sensitivity calculates the sensitivity to all covariates, established covariates, and unestablished covariates
#for each replicate
sensitivity <- function(x){
  small_number = .Machine$double.eps^0.5
  mycoefs <- x[[1]] #extracting coefficients and removing intercept
  if (is.null(names(mycoefs))==0){
    if (names(mycoefs)[1]=="(Intercept)"){
      mycoefs <- mycoefs[2:(p1+p2+1)]
    }
  }  
  
  thechosen <- which(abs(mycoefs) > small_number) #identifying which coefficients were selected--i.e. not zero
  
  truesigs <- length(setdiff(thechosen,falseind)) #identifying which of the selected coefficients are actually non-zero
  trueestsigs <- length(setdiff(thechosen,falseunind)) #identifying established, selected, truly non-zero coefficients
  trueunestsigs <- length(setdiff(thechosen,falseestind)) #identifying unestablished, selected, truly non-zero coefficients
  
  mysens <- truesigs/length(trueind) #calculating overall sensitivity 
  mysensE <- trueestsigs/length(trueeind) #calculating sensitivity to established covariates
  mysensU <- trueunestsigs/length(trueuind) #calculating sensitivity to unestablished covariates
  
  results <- c(mysens, mysensE,mysensU)
  return(results)
}

#findtruesigs returns the number of selected covariates that are actually non-zero
findtruesigs <- function(x){
  small_number = .Machine$double.eps^0.5
  mycoefs <- x[[1]] #extracting coefficients and removing intercept
  if (is.null(names(mycoefs))==0){
    if (names(mycoefs)[1]=="(Intercept)"){
      mycoefs <- mycoefs[2:(p1+p2+1)]
    }
  }  
  
  thechosen <- which(abs(mycoefs) > small_number) #identifying which coefficients were selected--i.e. not zero
  truesigs <- length(setdiff(thechosen,falseind))
  
  return(truesigs)
}

#findtruesigs_u returns the number of selected, UNESTABLISHED covariates that are actually non-zero
findtruesigs_u <- function(x){
  small_number = .Machine$double.eps^0.5
  mycoefs <- x[[1]] #extracting coefficients and removing intercept
  if (is.null(names(mycoefs))==0){
    if (names(mycoefs)[1]=="(Intercept)"){
      mycoefs <- mycoefs[2:(p1+p2+1)]
    }
  }  
  
  thechosen <- which(abs(mycoefs) > small_number) #identifying which coefficients were selected--i.e. not zero
  truesigs_u <- length(setdiff(thechosen,falseestind)) #identifying unestablished, selected, truly non-zero coefficients
  
  return(truesigs_u)
}

#findselsigs returns the number of covariates selected
findselsigs <- function(x){
  small_number = .Machine$double.eps^0.5
  mycoefs <- x[[1]] #extracting coefficients and removing intercept
  if (is.null(names(mycoefs))==0){
    if (names(mycoefs)[1]=="(Intercept)"){
      mycoefs <- mycoefs[2:(p1+p2+1)]
    }
  }  
  
  thechosen <- which(abs(mycoefs) > small_number) #identifying which coefficients were selected--i.e. not zero
  selsigs <- length(thechosen)
  return(selsigs)
}

#findselsigs_u returns the number of unestablished covariates selected
findselsigs_u <- function(x){
  small_number = .Machine$double.eps^0.5
  mycoefs <- x[[1]] #extracting coefficients and removing intercept
  if (is.null(names(mycoefs))==0){
    if (names(mycoefs)[1]=="(Intercept)"){
      mycoefs <- mycoefs[2:(p1+p2+1)]
    }
  }  
  
  thechosen <- which(abs(mycoefs) > small_number) #identifying which coefficients were selected--i.e. not zero
  selsigs_u <- length(setdiff(thechosen, which_set1))
  return(selsigs_u)
}

#numbetas counts the number of betas selected by the method and outputs them. It 
#can be used on any of the methods
numbetas <- function(x){
  mycoefs <- x[[1]] #extracting coefficients and removing intercept
  if (is.null(names(mycoefs))==0){
    if (names(mycoefs)[1]=="(Intercept)"){
      mycoefs <- mycoefs[2:(p1+p2+1)]
    }
  }  
  betas <- length(which(mycoefs != 0))
  return(betas)
}

