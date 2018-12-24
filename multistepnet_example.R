####################################################################################################

## File name: multipstepnet_example.R
## Programmer: Elizabeth Chase
## Project: Multi-step elastic net, in collaboration with Phil Boonstra
## Date: Worked on from Oct. 1, 2017-June 1, 2018; then more revisions for upload to GitHub throughout
##       fall 2018.
## Other related files:
## Input needed: imputedData_all.RData 
## Purpose: This file cleans MICE-imputed ECMO data from the PED-Rescuers prediction model and then 
##          performs an elastic net, underpenalized elastic net, multi-step net, IPF-Lasso, and sparse
##          group lasso. It processes the output across imputations to obtain AUC, Brier score, and 
##          the distribution of beta coefficients for each method. 

####################################################################################################

#Loading needed packages:
library(mix);
library(mice);
library(glmnet);
library(ipflasso);
library(SGL);
library(parallel);
library(pROC);
library(ggplot2);

setwd("~/Desktop/Research/Phil Elastic Net/Data Example 2");

source("~/Desktop/Research/Phil Elastic Net/GitHub Work/MSexample_functions_revised.R");

#Load imputed data:
load(paste0("imputedData_all.RData"));
dat_obs = all_dat;

#Initiate cluster
no_cores <- detectCores()-1
cl <- makeCluster(no_cores);

#Now we will tidy up the imputed data. First, we label the outcome and established/unestablished covariates:
outcome_1died = "outcome_1died";
est_covariates = orig_covariates[!orig_covariates%in%c("MAP","VentType","MAP_Other")];
unest_covariates = aug_covariates[!aug_covariates%in%c("pO21","FiO21","WBC","WBC_low","WBC_high","Platelets","Platelets_low")];

#Now, we calculate a global standard based on the means and standard deviations of the observed values,
#which we use for standardization. We do the established covariates first, then unestablished, and then
#mean airway pressure values (these have to be done separately):
#Established covariates:
x_obs = dat_obs[,est_covariates];
col_classes_est = unlist(lapply(x_obs,class));
full_rank_est = !names(col_classes_est[which(col_classes_est=="factor")])%in%c("DX0P");
x_obs = cbind(x_obs[,which(col_classes_est!="factor")],as.dummy(x_obs[,which(col_classes_est=="factor")],full_rank_est));
x_obs = x_obs[,grep("DX0POther",colnames(x_obs),invert=T)];
p = ncol(x_obs);

#Unestablished covariates
x_obs_unest = dat_obs[,unest_covariates];
col_classes_unest = unlist(lapply(x_obs_unest,class));
full_rank_unest = rep(T,sum(col_classes_unest=="factor"));
x_obs_unest = cbind(x_obs_unest[,which(col_classes_unest!="factor")],as.dummy(x_obs_unest[,which(col_classes_unest=="factor"),drop=F],full_rank_unest));
x_obs = cbind(x_obs,x_obs_unest);
rm(x_obs_unest);
q = ncol(x_obs) - p;
x_obs = data.matrix(x_obs);

#Standardize all covariates
mean_x_obs = colMeans(x_obs,na.rm=T);
n_obs = colSums(!is.na(x_obs));
sd_x_obs = sqrt(colSums(x_obs^2,na.rm=T)-colSums(x_obs,na.rm=T)^2/n_obs)/sqrt(n_obs-1);

#Now redoing mean airway pressure standardization:
for(vent_type in c("Conv","HFOV")) {
  curr_index = grep(vent_type,dat_obs$VentType);
  curr_column = paste0("MAP_",vent_type);
  curr_index2 = curr_index[which(!is.na(x_obs[curr_index,curr_column]))];
  curr_sum = sum(x_obs[,curr_column],na.rm=T);
  curr_sumsq = sum(x_obs[,curr_column]^2,na.rm=T);
  curr_sd = sqrt(curr_sumsq-curr_sum^2/length(curr_index2))/sqrt(length(curr_index2)-1);
  mean_x_obs[curr_column] = curr_sum/length(curr_index2);
  sd_x_obs[curr_column] = curr_sd;
}

est_covariates_before_dummy = est_covariates;
est_covariates = names(mean_x_obs)[1:p];
unest_covariates_before_dummy = unest_covariates;
unest_covariates = names(mean_x_obs)[(p+1):(p+q)];

#Now labelling our outcome as y
y <- dat_obs$outcome_1died

n_cv_rep = 25;

#We make a test dataset (testx) using the 26th imputation; we standardize it as above:
foo <- mice::complete(imputations_all,26)[,c(est_covariates_before_dummy,unest_covariates_before_dummy,"VentType")];
x_imp_est = foo[,est_covariates_before_dummy];
x_imp_est = cbind(x_imp_est[,which(col_classes_est!="factor")],as.dummy(x_imp_est[,which(col_classes_est=="factor")],full_rank_est));
x_imp_est = x_imp_est[,grep("DX0POther",colnames(x_imp_est),invert=T)];

x_imp_unest = foo[,unest_covariates_before_dummy];
x_imp_unest = cbind(x_imp_unest[,which(col_classes_unest!="factor")],as.dummy(x_imp_unest[,which(col_classes_unest=="factor"),drop=F],full_rank_unest));
x = cbind(x_imp_est,x_imp_unest);

x = data.matrix(x);
stopifnot(colnames(x)==colnames(x_obs));
stopifnot(colnames(x)==names(mean_x_obs));
stopifnot(colnames(x)==names(sd_x_obs));
stopifnot(sort(unique(as.numeric(x-x_obs)))==0);

testx = scale(x,center=mean_x_obs,scale=sd_x_obs);
testx[foo$VentType!="HFOV",grep("_HFOV",colnames(testx))] = 0;
testx[foo$VentType!="Conv",grep("_Conv",colnames(testx))] = 0;
testdat <- cbind(testx, y)

#In order to perform our functions across the 25 imputations, we need to assemble the 
#standardized imputed datasets as a list. For each of 25 imputations, we standardize it
#as above:
imps <- as.list(matrix(NA,1,n_cv_rep))
for (m in 1:n_cv_rep){
  foo = mice::complete(get(paste0("imputations_all")),m)[,c(est_covariates_before_dummy,unest_covariates_before_dummy,"VentType")];
  x_imp_est = foo[,est_covariates_before_dummy];
  x_imp_est = cbind(x_imp_est[,which(col_classes_est!="factor")],as.dummy(x_imp_est[,which(col_classes_est=="factor")],full_rank_est));
  x_imp_est = x_imp_est[,grep("DX0POther",colnames(x_imp_est),invert=T)];
  
  x_imp_unest = foo[,unest_covariates_before_dummy];
  x_imp_unest = cbind(x_imp_unest[,which(col_classes_unest!="factor")],as.dummy(x_imp_unest[,which(col_classes_unest=="factor"),drop=F],full_rank_unest));
  x = cbind(x_imp_est,x_imp_unest);
  
  x = data.matrix(x);
  stopifnot(colnames(x)==colnames(x_obs));
  stopifnot(colnames(x)==names(mean_x_obs));
  stopifnot(colnames(x)==names(sd_x_obs));
  stopifnot(sort(unique(as.numeric(x-x_obs)))==0);
  
  std_all_x = scale(x,center=mean_x_obs,scale=sd_x_obs);
  std_all_x[foo$VentType!="HFOV",grep("_HFOV",colnames(std_all_x))] = 0;
  std_all_x[foo$VentType!="Conv",grep("_Conv",colnames(std_all_x))] = 0;
  
  traindat <- cbind(std_all_x, y)
  imps[[m]] <- rbind(traindat, testdat)
}

#Now we will set our penalized regression parameters: the alpha sequence and number of folds 
alpha_seq = c(0, 0.1, 0.2)
n_folds = 5;

#Now running the netimp function on the 25 imputations:
set.seed(12192018)

mynet_data <- mclapply(imps,netimp,n_train = 178, p1=11, p2=11, alpha_seq=alpha_seq, n_folds=n_folds,
                       mc.preschedule=TRUE,mc.set.seed=FALSE,mc.silent=TRUE, mc.cores=no_cores,mc.cleanup=TRUE)

#And finally the sglimp function on the 25 imputations:
mysgl_data <- mclapply(imps,sglimp,n_train = 178, p1=11, p2=11, n_folds=n_folds, mc.preschedule=TRUE,
                       mc.set.seed=FALSE,mc.silent=TRUE, mc.cores=no_cores,mc.cleanup=TRUE)

##Now evaluating output:

#We will extract the best models:
en <- mclapply(mynet_data,subnet, quelmod="en", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
               mc.cores=no_cores,mc.cleanup=TRUE)

ipfen <- mclapply(mynet_data,subnet, quelmod="ipf_en", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
               mc.cores=no_cores,mc.cleanup=TRUE)

ms <- mclapply(mynet_data,subnet, quelmod="ms", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
               mc.cores=no_cores,mc.cleanup=TRUE)

ipflasso <- mclapply(mynet_data,sublasso, mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
               mc.cores=no_cores,mc.cleanup=TRUE)

sgl <- mclapply(mysgl_data,subsgl, quelmod="sgl", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
               mc.cores=no_cores,mc.cleanup=TRUE)

glasso <- mclapply(mysgl_data,subsgl, quelmod="group", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
               mc.cores=no_cores,mc.cleanup=TRUE)

#And now getting Brier Scores:
enBrier <- mean(unlist(mclapply(en, getassess, quelmod="brier", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)))

ipfenBrier <- mean(unlist(mclapply(ipfen, getassess, quelmod="brier", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                                 mc.cores=no_cores,mc.cleanup=TRUE)))

msBrier <- mean(unlist(mclapply(ms, getassess, quelmod="brier", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)))

ipflassoBrier <- mean(unlist(mclapply(ipflasso, getassess, quelmod="brier", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                                    mc.cores=no_cores,mc.cleanup=TRUE)))

sglBrier <- mean(unlist(mclapply(sgl, getassess, quelmod="brier", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                               mc.cores=no_cores,mc.cleanup=TRUE)))

glassoBrier <- mean(unlist(mclapply(glasso, getassess, quelmod="brier", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                                  mc.cores=no_cores,mc.cleanup=TRUE)))

#And now getting the AUC:
enAUC <- mean(unlist(mclapply(en, getassess, quelmod="auc", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                               mc.cores=no_cores,mc.cleanup=TRUE)))

ipfenAUC <- mean(unlist(mclapply(ipfen, getassess, quelmod="auc", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)))

msAUC <- mean(unlist(mclapply(ms, getassess, quelmod="auc", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)))

ipflassoAUC <- mean(unlist(mclapply(ipflasso, getassess, quelmod="auc", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)))

sglAUC <- mean(unlist(mclapply(sgl, getassess, quelmod="auc", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)))

glassoAUC <- mean(unlist(mclapply(glasso, getassess, quelmod="auc", mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)))

myauc <- c(enAUC, ipfenAUC, msAUC, ipflassoAUC, sglAUC, glassoAUC)
mybrier <- c(enBrier, ipfenBrier, msBrier, ipflassoBrier, sglBrier, glassoBrier)
methods <- c("EN", "IPFEN", "MS", "IPFLasso", "SGL", "GLASSO")
Performance <- data.frame("Method" = methods, "AUC" = myauc, "Brier" = mybrier)
save(Performance, file = "Performance.Rda")

#We will obtain the betas
enbeta <- mclapply(en, getbeta, mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)

ipfenbeta <- mclapply(ipfen, getbeta, mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                                 mc.cores=no_cores,mc.cleanup=TRUE)

msbeta <- mclapply(ms, getbeta, mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                              mc.cores=no_cores,mc.cleanup=TRUE)

ipflassobeta <- mclapply(ipflasso, getbeta, mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                                    mc.cores=no_cores,mc.cleanup=TRUE)

sglbeta <- mclapply(sgl, getbeta, mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                               mc.cores=no_cores,mc.cleanup=TRUE)

glassobeta <- mclapply(glasso, getbeta, mc.preschedule=TRUE,mc.set.seed=TRUE,mc.silent=TRUE,
                                  mc.cores=no_cores,mc.cleanup=TRUE)

#Now we will get the data in the form that we want for GGplot, namely, long form:
Dat <- matrix(NA, nrow=0, ncol=3)

for (i in 1:22){
  ENBeta <- unlist(sapply(enbeta, "[[", i))
  Variable <- rep(i, 25)
  Method <- rep('EN',25) 
  DatEN <- data.frame("Variable" = Variable, "Method" = Method, "Effect" = ENBeta)
  
  IPFENBeta <- unlist(sapply(ipfenbeta, "[[", i))
  Variable <- rep(i,25)
  Method <- rep('IPF-EN',25) 
  DatIPFEN <- data.frame("Variable" = Variable, "Method" = Method, "Effect" = IPFENBeta)
  
  IPFLBeta <- unlist(sapply(ipflassobeta, "[[", i))
  Variable <- rep(i, 25)
  Method <- rep('IPF-Lasso', 25)
  DatIPFL <- data.frame("Variable" = Variable, "Method" = Method, "Effect" = IPFLBeta)
  
  MSBeta <- unlist(sapply(msbeta, "[[", i))
  Variable <- rep(i,25)
  Method <- rep('MS',25) 
  DatMS <- data.frame("Variable" = Variable, "Method" = Method, "Effect" = MSBeta)
  
  GLASBeta <- unlist(sapply(glassobeta, "[[", i))
  Variable <- rep(i,25)
  Method <- rep('GLASSO',25) 
  DatGLAS <- data.frame("Variable"= Variable, "Method"= Method, "Effect" = GLASBeta)
 
  SGLBeta <- unlist(sapply(sglbeta, "[[", i))
  Variable <- rep(i,25)
  Method <- rep('SGL',25) 
  DatSGL <- data.frame("Variable" = Variable, "Method" = Method, "Effect" = SGLBeta)
  
  Dat <- rbind(Dat, DatEN, DatIPFEN, DatIPFL, DatMS, DatSGL, DatGLAS)
}

original <- c(rep(0,1650), rep(1,1650))
Dat <- cbind(Dat, original)
names(Dat) <- c("Variable", "Method", "Effect","Category")

#I created the variable labels "by hand", one at a time, because everything
#was so customized. 
names_betas_to_plot = vector("character",p+q);
names_betas_to_plot[1] = "paste(admit~hours~pre-ECMO~(log))";    
names_betas_to_plot[2] = "paste(intubated~hours~pre-ECMO~(log))";  
names_betas_to_plot[3] = "paste(pH)";  
names_betas_to_plot[4] = "paste(PaCO[2],', ',mm~Hg)";  
names_betas_to_plot[5] = "paste(MAP~(CMV),', ',cm~H[2]*O)";
names_betas_to_plot[6] = "paste(MAP~(HFOV),', ',cm~H[2]*O)";
names_betas_to_plot[7] = "paste(malignancy)";
names_betas_to_plot[8] = "paste(DX,': ',Asthma)";  
names_betas_to_plot[9] = "paste(DX,': ',Bronchiolitis)";
names_betas_to_plot[10] = "paste(DX,': ',Pertusis)";
names_betas_to_plot[11] = "paste(preECMO~milrinone)";
names_betas_to_plot[12] = "paste(bilirubin~mg,'/',dL~(log))";  
names_betas_to_plot[13] = "paste(ALT~U,'/',L~(log))";
names_betas_to_plot[14] = "paste(extent~of~leukocytosis~(log))";#excess~WBC>15%*%10^9/L~(log)";
names_betas_to_plot[15] = "paste(extent~of~leukopenia~(log))";#deficient~WBC<5%*%10^9/L~(log)";
names_betas_to_plot[16] = "paste(extent~of~thrombocytopenia~(log))";#deficient~platelets<150%*%10^9/L~(log)";
names_betas_to_plot[17] = "paste(INR)";
names_betas_to_plot[18] = "paste(VIS~(log))";
names_betas_to_plot[19] = "paste(lactate~mMol,'/',L~(log))";
names_betas_to_plot[20] = "paste(PF~ratio~(log))";
names_betas_to_plot[21] = "paste(abnormal~pupillary~response)";
names_betas_to_plot[22] = "paste(preECMO~acute~kidney~injury)";

Dat[,"Variable"] = factor(Dat[,"Variable"], 
                          levels = c(1:22), 
                          ordered = T, 
                          labels = names_betas_to_plot);


#Dat should be a matrix with 4 columns (Variable, Method, Effect, and Category) and 3300
#observations: 22 betas x 25 imputations x 6 methods = 3300
save(Dat,file="RealData_Betas.Rda")
load("RealData_Betas.Rda")

aggregate(Dat$Effect, list(Dat$Variable, Dat$Method), mean)