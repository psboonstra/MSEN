###################################################################################################

## File name: MSexample_functions_revised.R
## Programmer: Elizabeth Chase
## Project: Multi-step elastic net, in collaboration with Phil Boonstra
## Date: Worked on from Oct. 1, 2017-June 1, 2018; this polished code was completed and assembled
##       for publication during fall 2018
## Other related files: multistepnet_example.R, multistepnet_functions_revised.R
## Purpose: This file contains functions to clean the ECMO Ped-Rescuers data; perform an elastic net,
##          multi-step elastic net, IPF-Lasso, and sparse group lasso; and analyze results from the 
##          above functions.

###################################################################################################

## PART 1: FUNCTIONS TO CLEAN IMPUTED ECMO DATA

# as.dummy takes the raw, imputed ECMO data and recodes categorical variables as dummy variables:
as.dummy = function(x,full_rank=T) {
  single.as.dummy <- function(x,full_rank) {
    levels_x = levels(x);
    1*matrix(rep(x,nlevels(x)) == rep(levels_x,each=length(x)),nrow=length(x),ncol=nlevels(x),dimnames=list(NULL,levels_x))[,(1+full_rank):length(levels_x),drop=F];
  }
  if("factor"%in%class(x)) {
    if(length(full_rank)>1) {warning("ignoring all but first element of 'full_rank'");}
    result = single.as.dummy(x,full_rank[1]);
  } else if("logical"%in%class(x)) {
    if(length(full_rank)>1) {warning("ignoring all but first element of 'full_rank'");}
    result = single.as.dummy(factor(x,levels=c(F,T)),full_rank[1]);
  } else if("integer"%in%class(x)) {
    if(length(full_rank)>1) {warning("ignoring all but first element of 'full_rank'");}
    result = single.as.dummy(factor(x,levels=sort(unique(x),decreasing = T)),full_rank[1]);
  } else if(class(x)=="data.frame") {
    result = NULL;
    full_rank = rep(full_rank,length=ncol(x));
    for(i in 1:ncol(x)) {
      if("factor"%in%class(x[,i])) {
        foo = single.as.dummy(x[,i],full_rank[i]);
        colnames(foo) = paste0(colnames(x)[i],colnames(foo));
        result = cbind(result,foo);
      } else if("logical"%in%class(x[,i])) {
        foo = single.as.dummy(factor(x[,i],levels=c(F,T)),full_rank[i]);
        colnames(foo) = paste0(colnames(x)[i],colnames(foo));
        result = cbind(result,foo);
      } else if("integer"%in%class(x[,i])) {
        foo = single.as.dummy(factor(x[,i],levels=sort(unique(x[,i]),decreasing = T)),full_rank[i]);
        colnames(foo) = paste0(colnames(x)[i],colnames(foo));
        result = cbind(result,foo);
      } else {
        stop("x must be either a factor, logical, or a dataframe comprised of factors/logicals/integers");
      }
    } 
  } else {
    stop("x must be either a factor, logical, or a dataframe comprised of factors/logicals/integers");
  }
  data.frame(result);
}

#expit is needed for the logistic regression setting
expit = function(x) {
  1/(1+exp(-x))
}

#is.integer0 is needed to identify empty vectors and lists
is.integer0 <- function(x)
{is.integer(x) && length(x) == 0L}

#netimp fits an elastic net, IPF-EN, IPF-Lasso, and MS to the imputed data; it does not
#consolidate across cross-validations to deal with the imputations and it does not 
#standardize the data because we have pre-standardized it

netimp <- function(dat, n_train, p1, p2, alpha_seq, n_folds){
  stopifnot(p1 + p2 == ncol(dat) - 1);
  stopifnot(n_train < nrow(dat));
  #We separate out the design matrix and outcome vector:
  n_test = nrow(dat) - n_train;
  
  y_train <- factor(1 * dat[1:n_train,"y"]);
  std_x_train <- dat[1:n_train, which(colnames(dat) != "y"),drop=F];
  std_x_test <- dat[n_train + (1:n_test), which(colnames(dat) != "y"),drop=F];
  y_test <- drop(dat[n_train + (1:n_test), which(colnames(dat) == "y"),drop=F]);
  
  #We now fit our 4 differently-penalized models for each of our 3 values of alpha for each of our 
  #cross-validated replicates:
  #Model 1: phi_1 = 0, phi_2 = 1 (no penalization on established and full on unestablished)
  penalties = rbind(penalty1 = c(rep(0,p1),rep(1,p2)),
                    #Model 2 (phi_1 = 1/16, phi_2 = 1, or small penalization on the established covariates)
                    penalty2 = c(rep(1/16,p1),rep(1,p2)),
                    #Model3 (Phi_1 = 1/2, Phi_2 = 1, or double penalization on the unestablished covariates)
                    penalty3 = c(rep(1/2,p1),rep(1,p2)),
                    #Model 4: phi_1 = 1, phi_2 = 1, or no penalization on the established covariates
                    penalty4 = c(rep(1,p1),rep(1,p2)));
  penalty_exclude = 
    cbind(ipf_en = c(Inf, 1, 1, 1), 
          en = c(Inf, Inf, Inf, 1), 
          ms = c(1, 1, 1, 1));
  n_penalties = nrow(penalties);
  
  #First, we initialize values to hold the deviance and lambda sequence for each
  #combination of alpha and phi (3 alphas and 5 phis)
  n_alphas = length(alpha_seq);
  store_dev = store_dev_lasso = 
    store_lambda_seq_lasso = store_lambda_seq = vector("list",n_penalties);
  
  for(k in 1:n_penalties) {#initialize values
    store_dev[[k]] = 
      store_lambda_seq[[k]] = vector("list",n_alphas);
    store_dev_lasso[[k]] = store_lambda_seq_lasso[[k]] = vector("list", 1);
    store_dev[[k]][1:n_alphas] = 0;
    store_dev_lasso[[k]][1] = 0;
    names(store_dev[[k]]) = 
      names(store_lambda_seq[[k]]) = alpha_seq;
  }
  
  #Because glmnet allows us to assign folds, we assign observations to folds to ensure
  #consistency across imputations. 
  foldid = sample(rep(1:n_folds,length = n_train));

  for(k in 1:n_penalties) {
      for(j in 1:n_alphas) {
        curr_fit = cv.glmnet(x = std_x_train, 
                             y = y_train,
                             standardize = F,
                             family = "binomial",
                             alpha = alpha_seq[j],
                             foldid = foldid,
                             lambda = store_lambda_seq[[k]][[j]],
                             penalty.factor = penalties[k,],
                             keep = T);
        store_dev[[k]][[j]] = store_dev[[k]][[j]] + curr_fit$cvm;
        if(is.null(store_lambda_seq[[k]][[j]])) {store_lambda_seq[[k]][[j]] = curr_fit$lambda;}
        #assign(paste0("alpha",j,"_model",k,"_fitted_prob"), get(paste0("fitted",k,"_mod",j))$fit.preval[,1:length(get(paste0("alpha",j,"_model",k,"_lambda_seq")))]);
      } 
      
      lasso_fit = cv.glmnet(x = std_x_train, 
                            y = y_train,
                            standardize = F,
                            family = "binomial",
                            alpha = 1,
                            foldid = foldid,
                            lambda = store_lambda_seq_lasso[[k]][[1]],
                            penalty.factor = penalties[k,],
                            keep = T);
      store_dev_lasso[[k]][[1]] = store_dev_lasso[[k]][[1]] + lasso_fit$cvm;
      if(is.null(store_lambda_seq_lasso[[k]][[1]])) {store_lambda_seq_lasso[[k]][[1]] = lasso_fit$lambda;} 
      
    cat(k,"\n");
  }
  
  which_best_alpha = apply(matrix(rapply(store_dev, min), nrow = n_penalties, byrow = T), 1, which.min);
  which_best_alpha_lasso = apply(matrix(rapply(store_dev_lasso, min), nrow = n_penalties, byrow = T), 1, which.min);
  best_lambda_seq = mapply("[[",store_lambda_seq, which_best_alpha); 
  if (class(best_lambda_seq)=="matrix"){
    best_lambda_seq = as.list(data.frame(best_lambda_seq));
  }
  best_lambda_seq_lasso = mapply("[[",store_lambda_seq_lasso,which_best_alpha_lasso);
  if (class(best_lambda_seq_lasso)=="matrix"){
    best_lambda_seq_lasso = as.list(data.frame(best_lambda_seq_lasso));
  }
  best_dev = mapply("[[",store_dev,which_best_alpha);
  if (class(best_dev)=="matrix"){
    best_dev = as.list(data.frame(mapply("[[",store_dev,which_best_alpha)));
  }
  best_dev_lasso = mapply("[[",store_dev_lasso,which_best_alpha_lasso);
  if (class(best_dev_lasso)=="matrix"){
    best_dev_lasso = as.list(data.frame(mapply("[[",store_dev_lasso,which_best_alpha_lasso)));
  }
  which_best_lambda = unlist(lapply(best_dev, which.min));
  which_best_lambda_lasso = unlist(lapply(best_dev_lasso, which.min));
  best_lambda = mapply("[", best_lambda_seq, which_best_lambda);
  best_lambda_lasso = mapply("[",best_lambda_seq_lasso, which_best_lambda_lasso)
  
  #And now we fit the final models using our optimal values of lambda and alpha for each penalty type:
  store_coefs = coefs_lasso = matrix(0, nrow = n_penalties, ncol = p1 + p2, dimnames = list(rownames(penalties), NULL));
  store_fits = fits_lasso = matrix(0, nrow = n_penalties, ncol = n_test, dimnames = list(rownames(penalties), NULL));
  store_assess = assess_lasso = matrix(0, nrow = n_penalties, ncol = 2, dimnames = list(rownames(penalties), c("brier","auc")));
  for(k in 1:n_penalties) {
    curr_fit = glmnet(x = std_x_train, 
                      y = y_train,
                      standardize = F,
                      family = "binomial",
                      alpha = alpha_seq[which_best_alpha[k]],
                      lambda = best_lambda_seq[[k]],
                      penalty.factor = penalties[k,]);
    
    lasso_fit = glmnet(x = std_x_train, 
                       y = y_train,
                       standardize = F,
                       family = "binomial",
                       alpha = 1,
                       lambda = best_lambda_seq_lasso[[k]],
                       penalty.factor = penalties[k,]);
    
    store_coefs[k,] = coef(curr_fit)[-1, which_best_lambda[k]];
    coefs_lasso[k,] = coef(lasso_fit)[-1,which_best_lambda_lasso[k]];
    store_fits[k,] = drop(predict(curr_fit, std_x_test, s = best_lambda[k], type="response"));
    fits_lasso[k,] = drop(predict(lasso_fit, std_x_test, s = best_lambda_lasso[k], type="response"));
    store_assess[k,"brier"] = mean((y_test - store_fits[k,])**2);
    store_assess[k,"auc"] = roc(response = y_test, predictor = store_fits[k,], smooth=FALSE, auc=TRUE, ci = FALSE, plot=FALSE)$auc;
    assess_lasso[k,"brier"] = mean((y_test - fits_lasso[k,])**2);
    assess_lasso[k,"auc"] = roc(response = y_test, predictor = fits_lasso[k,], smooth=FALSE, auc=TRUE, ci = FALSE, plot=FALSE)$auc;
  }
  
  #Finally, we determine which of the five models is best overall for each penalty combination:
  selected_penalties_1 = lapply(best_dev, min);
  selected_penalties_2 = penalty_exclude * unlist(selected_penalties_1); 
  selected_penalties = apply(selected_penalties_2, 2, which.min);
  
  penalty_lasso = which.min(unlist(lapply(best_dev_lasso,min))[2:4])+1;
  tuning_par = cbind(alpha = alpha_seq[which_best_alpha], 
                     lambda = best_lambda);
  rownames(tuning_par) = rownames(store_coefs);
  tuning_par_lasso = cbind(alpha = 1, 
                           lambda = best_lambda_lasso);
  rownames(tuning_par_lasso) = rownames(store_coefs);
  
  return(list(setup = list(dat = dat, n_train = n_train, p1 = p1, p2 = p2, alpha_seq = alpha_seq, n_folds = n_folds), 
              selected_penalties = selected_penalties,
              store_coefs = store_coefs, 
              store_fits = store_fits,
              store_assess = store_assess,
              tuning_par = tuning_par,
              ipflasso = list(penalty = penalty_lasso, coefs = coefs_lasso, fits = fits_lasso, assess = assess_lasso, tuning_par = tuning_par_lasso)));
  
}

#netsgl fits a sparse group lasso and group lasso to the imputed data; it does not
#consolidate across cross-validations to deal with the imputations

sglimp <- function(dat, n_train, p1, p2, n_folds){
  
  stopifnot(p1 + p2 == ncol(dat) - 1);
  stopifnot(n_train < nrow(dat));
  #We separate out the design matrix and outcome vector:
  n_test = nrow(dat) - n_train;
  
  y_train <- dat[1:n_train,"y"];
  std_x_train <- dat[1:n_train, which(colnames(dat) != "y"),drop=F];
  std_x_test <- dat[n_train + (1:n_test), which(colnames(dat) != "y"),drop=F];
  y_test <- drop(dat[n_train + (1:n_test), which(colnames(dat) == "y"),drop=F]);
  
  mydat = list(x = std_x_train, y = y_train)
  
  #We initialize values to store the deviance and lambda sequence 
  modelsgl_dev = 
    modelgroup_dev = 0
  modelsgl_lambda_seq = 
    modelgroup_lambda_seq = NULL;
  
  #We create a vector to tell SGL which group each covariate belongs to:
  myindex <- c(rep(1,p1),rep(2,p2));
  
  #First fitting a sparse group lasso with alpha = 0.95:
    fitted_sgl =
      cvSGL(data = mydat, 
            index = myindex, 
            type = "logit",
            standardize = F,
            alpha = 0.95,
            nlam = 100,
            nfold = n_folds, 
            lambdas = modelsgl_lambda_seq);
    modelsgl_dev = modelsgl_dev + fitted_sgl$lldiff;
    if(is.null(modelsgl_lambda_seq)) {modelsgl_lambda_seq = fitted_sgl$fit$lambdas;}
    
  #And now fitting a grouped lasso with alpha = 0:
    fitted_group = 
      cvSGL(data = mydat, 
            index = myindex, 
            type = "logit",
            standardize = F,
            alpha = 0,
            nlam = 100,
            nfold = n_folds,
            lambdas = modelgroup_lambda_seq);
    modelgroup_dev = modelgroup_dev + fitted_group$lldiff;
    if(is.null(modelgroup_lambda_seq)) {modelgroup_lambda_seq = fitted_group$fit$lambdas;}
    
  #We locate the best lambda value for each model
  sgl_which_lambda = which.min(modelsgl_dev);
  sgl_lambda_select = modelsgl_lambda_seq[sgl_which_lambda];
  
  group_which_lambda = which.min(modelgroup_dev);
  group_lambda_select = modelgroup_lambda_seq[group_which_lambda];
  
  #Now we fit the final models, using our best values of alpha and lambda:
  store_coefs = matrix(0, nrow = 2, ncol = p1 + p2, dimnames = list(c("sgl","group"), NULL));
  store_fits = matrix(0, nrow = 2, ncol = n_test, dimnames = list(c("sgl","group"), NULL));
  store_assess = matrix(0, nrow = 2, ncol = 2, dimnames = list(c("sgl","group"), c("brier","auc")));
  
  fitted_sgl = SGL(data = mydat,
                   index = myindex, 
                   type = "logit",
                   standardize = F,
                   alpha = 0.95,
                   nlam = 100,#Weird thing is that even though we provide the 'modelsgl_lambda_seq', we still need to specify this length
                   lambdas = modelsgl_lambda_seq);
  
  store_coefs["sgl",] = fitted_sgl$beta[,sgl_which_lambda];
  store_fits["sgl",] = drop(predictSGL(fitted_sgl, std_x_test, sgl_which_lambda));
  store_assess["sgl","brier"] = mean((y_test - store_fits["sgl",])**2);
  store_assess["sgl","auc"] = roc(response = y_test, predictor = store_fits["sgl",], smooth=FALSE, auc=TRUE, ci = FALSE, plot=FALSE)$auc;
  
  fitted_group = SGL(data = mydat,
                     index = myindex, 
                     type = "logit",
                     standardize = F,
                     alpha = 0.95,
                     nlam = 100,#Weird thing is that even though we provide the 'modelgroup_lambda_seq', we still need to specify this length
                     lambdas = modelgroup_lambda_seq);
  
  store_coefs["group",] = fitted_group$beta[,group_which_lambda];
  store_fits["group",] = drop(predictSGL(fitted_group, std_x_test, group_which_lambda));
  store_assess["group","brier"] = mean((y_test - store_fits["group",])**2);
  store_assess["group","auc"] = roc(response = y_test, predictor = store_fits["group",], smooth=FALSE, auc=TRUE, ci = FALSE, plot=FALSE)$auc;
  
  tuning_par = cbind(alpha = c(0.95, 0),
                     lambda = c(sgl_lambda_select, group_lambda_select));
  rownames(tuning_par) = c("sgl","group");
  
  return(list(setup = list(dat = dat, n_train = n_train, p1 = p1, p2 = p2, n_folds = n_folds), 
              store_coefs = store_coefs, 
              store_fits = store_fits,
              store_assess = store_assess,
              tuning_par = tuning_par));
  
}

subnet <- function(x, quelmod){ 
  if (quelmod=="autozero") {
    penalty_pick = 1
  } else {penalty_pick = x$selected_penalties[quelmod]};
  
  return(list(setup = x$setup, 
              tuning_par = c(x$tuning_par[penalty_pick,],phi_cat = as.numeric(penalty_pick)),
              coefs = x$store_coefs[penalty_pick,],
              fits = x$store_fits[penalty_pick,],
              assess = x$store_assess[penalty_pick,]));
}

sublasso <- function(x){
  penalty_pick = x$ipflasso$penalty;
  
  return(list(setup = x$setup,
              tuning_par = c(x$ipflasso$tuning_par[penalty_pick,], phi_cat = as.numeric(penalty_pick)),
              coefs = x$ipflasso$coefs[penalty_pick,],
              fits = x$ipflasso$fits[penalty_pick,],
              assess = x$ipflasso$assess[penalty_pick,]));
}

#subsgl is used on the mysgl output to extract the different sgl penalties
subsgl <- function(x,quelmod) {
  
  return(list(setup = x$setup, 
              tuning_par = c(x$tuning_par[quelmod,]),
              coefs = x$store_coefs[quelmod,],
              fits = x$store_fits[quelmod,],
              assess = x$store_assess[quelmod,]));
}

getassess <- function(x, quelmod){
  return(x$assess[quelmod])
}

getbeta <- function(x){
  return(x$coefs)
}