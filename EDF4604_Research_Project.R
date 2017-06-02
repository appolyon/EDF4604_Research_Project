# Recreating Kosinsky's Analysis
# From paper 'Mining Big Data to Extract Patterns and Predict Real-Life Outcome'

# Install necessary packages
install.packages("foreign", dependencies=TRUE)
install.packages("Rcmdr", dependencies=TRUE)
install.packages("psych", dependencies=TRUE)
install.packages("Matrix", dependencies=TRUE) # From Kosinsky article
install.packages("irlba", dependencies=TRUE) # From Kosinsky article
install.packages("topicmodels", dependencies=TRUE) # From Kosinsky article
install.packages("ROCR", dependencies=TRUE) # From Kosinsky article

# Load packages
library(foreign)
library(Rcmdr)
library(psych)
library(Matrix) # From Kosinsky article
library(irlba) # From Kosinsky article
library(topicmodels) # From Kosinsky article
library(ROCR) # From Kosinsky article

# Description of files used in the analysis
# users.csv: contains psychodemographic user profiles. It has 110,728 rows
# (excluding the row holding column names) and nine columns: anonymized user ID,
# gender ("0" for male and "1" for female), age, political views ("0" for Democrat and "1" for Republican),
# and five scores on a 100-item-long International Personality Item Pool questionnaire measuring the five-factor
# (i.e., Openness, Conscientiousness, Extroversion, Agreeable-ness, and Neuroticism) model of personality

# likes.csv: contains anonymized IDs and names of nL ???
# 1,580,284 Facebook Likes. It has two columns: ID and name.

# users-likes.csv: contains the associations between users and their Likes, stored as user-Like pairs. It has nu-L
# 10,612,326 rows and two columns: user ID and Like ID. An existence of a user-Like pair implies that a given user
# had the corresponding Like on their profile.

# Set working directory to where you downloaded the original files
setwd("D:/Monash/EDF4604 Research project/01 Research Project/EDF4604_Research_Project")

# Load files into R using
users <- read.csv("users.csv")
likes <- read.csv("likes.csv")
ul <- read.csv("users-likes.csv")

# Inspect the dimensions of the data
# Inspect head of data
# Inspect tail of data
# Kosinsky only does this for ul

dim(users)
head(users)
tail(users)

dim(likes)
head(likes)
tail(likes)

dim(ul)
head(ul)
tail(ul)

# Constructing a user like matrix
# Matrix named M
# First need to match users with their likes

ul$user_row <- match(ul$userid, users$userid)
ul$like_row <- match(ul$likeid, likes$likeid)

# Use the pointers to rows in the 'users' and 'likes' objects
# to build a user-Like matrix

require(Matrix)
# M <- sparseMatrix(i = ul$user_row, j = ul$like_row, x = 1) # Original command
M <- sparseMatrix(i = ul$user_row, j = ul$like_row, x = 1)

# Set row names of the matrix to contain the iDs of the respective users
# Set column names to contain the names of the respective Likes
# Display dimensions of Matrix M

rownames(M) <- users$userid
colnames(M) <- likes$name
dim(M)

# Objects ul and likes are not necessary for further steps and can be removed
# Uncomment following line to remove the objects
# rm(ul, likes)

# Trimming the User-Like Matrix
# Remove the least frequent data points 
# Thresholds are 50 minimum likes per user and minimum 150 users per
# like to reduce the time required for further analysis

repeat {
  i <- sum(dim(M))
  M <- M[rowSums(M) >= 50, colSums(M) >= 150]
  if (sum(dim(M)) == i) break
}

users <- users[match(rownames(M), users$userid),]

# Extracting patterns from big data sets
# Singular value decomposition (SVD)
# Latent Dirichlet allocation (LDA)
# Reducing the dimensionality of the User-Like matrix using SVD and LDA
# Package used in this section (irlba) should have loaded at start
# Parameter nv = 5 corresponds to k, or the number of SVD dimensions to be extracted

# SVD
set.seed(seed = 68)
Msvd <- irlba(M, nv = 5)
u <- Msvd$u
v <- Msvd$v

# Note: We do not center the data prior to conducting the SVD to maintain the sparse 
# format of Matrix M
# Centering could be achieved by using the following command:
# M <- scale(M, scale = F)

# Scree plot representing the singular values of the consecutive 
# SVD dimensions (matrix SIGMA) can be displayed using:

plot(Msvd$d)

# Use varimax function to rotate the SVD dimensions
# The following code produces V_rot and U_rot, the 
# varimax rotated equivalents of matrices U and V:

v_rot <- unclass(varimax(Msvd$v)$loadings)
u_rot <- as.matrix(M %*% v_rot)

# LDA
# topicmodels package should install and load at beginning
# Set Dirichlet distribution parameters to a = 10 and delta = 0.1
# As with SVD, preset R's random number generator to seed = 68

Mlda <- LDA(M, k = 5, control = 
            list(alpha = 10, delta = .1, seed = 68), 
            method = "Gibbs")
gamma <- Mlda@gamma
beta <- exp(Mlda@beta)

# Following code is used to compute the log-likelihoods across different values of k
# The for loop cycles through increasing values of i, trains the LDA model while
# setting the k = 1, adn the extracts the model's log-likelihood and its degrees
# of freedom using teh logLik function. The results are saved in the lg object.
# Consider widening the range of k's to be tested but be aware that this code can 
# take a very long time to run depending on the computer it is run on

lg <- list()
for (i in 2:5) {
  Mlda <- LDA(M, k = i, control = 
  list(alpha = 10, delta = .1, seed = 68),
  method = "Gibbs")
  lg[[i]] <- logLik(Mlda)
}
plot(2:5, unlist(lg))

# Interpreting clusters and dimensions
# For simplicity, we will use k = 5 LDA clusters and k = 5 SVD dimensions 
# extracted from Matrix M, but you should experiment with different
# vallues of k 

# LDA clusters
# Start by computing the correlations between user scores on LDA clusters
# stored in R object 'gamma' and psychodemographic user traits
# Code exludes first column of 'users' because they are user IDs

cor(gamma, users[, - 1], use = "pairwise")

# Investigate which Likes are most representative of these k = 5 LDA clusters
# Strength of relationship stored in R object 'beta'
# Following code will extract top 10 likes most stongly associated 
# with each of the clusters
# First we create an empty list 'top' to store results
# Next we start a 'for' loop assigning consecutive values from 1 to 5 to i
# Inside the loop, we use 'order' function to order the LIkes in ascending 
# order based on their scores on the i-th LDA cluster. Next we use the 'tail' 
# function to extract teh indexes of the last 10 LIkes (i.e. the Likes with 
# the higest LDA scores) and save their names as teh i-th element of list 'top'

top <- list()
for (i in 1:5) {
  f <- order(beta[i,])
  temp <- tail(f, n = 10)
  top[[i]] <- colnames(M) [temp]
}
top

# SVD Dimensions

cor(u_rot, users[, -1], use = "pairwise")

# 'For' loop from LDA adapted for SVD

top <- list()
bottom <-list()
for (i in 1:5) {
  f <- order(v_rot[ ,i])
  temp <- tail(f, n = 10)
  top[[i]]<-colnames(M)[temp]  
  temp <- head(f, n = 10)
  bottom[[i]]<-colnames(M)[temp]  
}

# Extract the indexes of the Likes with extreme scores

colnames(M) [tail(f, n = 10)]
colnames(M) [head(f, n = 10)]

# Predicting real life outcomes
# Cross-validation
# Predicting real-life outcomes with Facebook likes
# Building a prediction model based on the SVD dimensions extracted
# from the user-Like Matrix M. Start by assigning a random number to 
# each user in the sample as a way of splitting users into 10 independent
# subsets (or folds) that will be used for cross validation

folds <- sample(1:10, size = nrow(users), replace = T)

# The users with fold = 1 are assigned to the test subset, and the remaining
# userse are assigned to the training subset. The following code produces a
# logical vector 'test' that takes the value of TRUE when object folds equals
# 1 and FALSE otherwise

test <- folds == 1

# Logical vectors can be used to extract desired elements from R objects, such 
# as other vectors or matrices. For example, the command mean(users$age[test]) 
# can be used to compute the average age of the users in the test subset.

# In the following step, we extract k = 50 SVD dimensions from a training
# subset of Facebook Likes (i.e. M[!test]). SVD scores of Likes are varimax-
# rotated user SVD scores of the entire sample. Consequently, the SVD scores
# for the users in the test subset, preserving the independence of the results
# obtained from the training and test subsets

Msvd <- irlba(M[!test,], nv = 50)
v_rot <- unclass(varimax(Msvd$v)$loadings)
u_rot <- as.data.frame(as.matrix(M %*% v_rot))

# Next apply the function 'glm' to build regression models predicting variables
# 'ope' (openess) and gender from the user SVD scores in the training subset
# Logistic regression model can be obtained by specifying the parameter 'family =
# "binomail""

fit_o <- glm(users$ope~., data = u_rot, subset = !test)

fit_g <- glm(users$gender~.,data = u_rot, subset = !test, family = "binomial")

# Next estimate the predictions for the testing sample using the function 'predict'
# Predictions are produced using the models developed in the previous step

pred_o <- predict(fit_o, u_rot[test,])

pred_g <- predict(fit_g, u_rot[test,], type = "response")

# Finally, estimate the accuracy of the predictions for this particular cross-
# validation fold. The accuracy of the linear predictions can be expressed
# as a Pearson product-momnent correlation

cor(users$ope[test], pred_o)

# For dichotomous variables such as gender, report prediction accuracy using 
# the area under the receiver-operating characteristic curve coefficient (AUC)
# which can be computed using the ROCR library

temp <- prediction(pred_g, users$gender[test])
performance(temp, "auc")@y.values

# So far we have estimated prediction performance based on all k = 50 SVD dimensions
# Next we will investigate the relationship between the cross-validated prediction 
# accuracy and the number of k of SVD dimensions. In the following code we first 
# define a set of ks containing the values of k that we want to test. We then create
# an empty list 'rs' that is used to store the results. Then we start a 'for' loop that
# reruns the code surrounded by curly brackets while changing the value of i to consecutive
# elements stored in ks.

set.seed(seed = 68)
ks <- c(2:10,15,20,30,40,50)
rs <- list()
for (i in ks) {
  v_rot <- unclass(varimax(Msvd$v[,1:i])$loadings)
  u_rot <- as.data.frame(as.matrix(M%*%v_rot))
  fit_o <- glm(users$ope~., data = u_rot, subset = !test)
  pred_o <- predict(fit_o, u_rot[test,])
  rs[[as.character(i)]] <- cor(users$ope[test], pred_o)
}

# Finally compute cross-validated predictions of openess for entire sample
# We build models and compute the predicted values using each of the 10 folds of data

set.seed(seed = 68)
pred_o <- rep(NA, n = nrow(users))
for (i in 1:10){
  test <- folds == i
  Msvd <- irlba(M[!test, ], nv = 50)
  v_rot <- unclass(varimax(Msvd$v)$loadings)
  u_rot <- as.data.frame(as.matrix(M %*% v_rot))
  fit_o <- glm(users$ope~., data = u_rot, subset = !test)
  pred_o[test] <- predict(fit_o, u_rot[test, ])
}
cor(users$ope, pred_o)


# Prediction models based on LDA cluster memberships

Mlda <- LDA(M[!test, ], control=list(alpha=1, delta=.1, seed=68), k=50, method="Gibbs")
temp<-posterior(Mlda, M)
gamma<-as.data.frame(temp$topics)
