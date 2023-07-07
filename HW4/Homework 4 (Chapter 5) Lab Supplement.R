# ISLR Chapter 5 Lab Supplement
# STAT 1361/2360:  Statistical Learning and Data Science
# University of Pittsburgh
# Prof. Lucas Mentch

# The lab provided in ISLR relies on built-in R functions to conduct cross-
# validation and bootstrapping and does not go through permutation tests.
# Most of these procedures are much more natural to simply program on your
# own so you can see each step in the procedure explicitly.  Here we'll walk through
# some examples.


####################################################################################
# Cross Validation 
####################################################################################
# First let's make up some data.  Here I'll generate a dataset with one covariate
# (also known as a predictor, feature, etc.) and let's say I want 998 observations:
x1 <- runif(998)

# This gives me 998 samples taken uniformly at random between 0 and 1.  Now I'll 
# generate 998 random errors and create a response that depends on both x1 and 
# x1^2 and store this in a data frame
eps <- rnorm(998,mean=0,sd=0.25)
y <- x1 + x1^2 + eps
df <- data.frame(y,x1,"x1sq"=x1^2)

# First let's fit a linear model and see if the squared term looks significant
# (it should -- that's the way we designed it!)
lm0 <- lm(y~.,data=df)

# the notation y~. tells R that I want to fit a linear model with the variable named
# 'y' treated as the response and everything else treated as a predictor.  Now let's 
# look at the summary:
summary(lm0)

# You should see that the squared term looks very significant.  But let's say we didn't
# trust this result.  Let's do 10-fold CV to determine whether we are predicting more
# accurately with the squared term in the model.  First we need to create an index that
# will serve to randomly shuffle the data.  The function sample will do this; sample (1:n)
# will provide a random permutation of the numbers 1, 2, 3, ..., n:
CVind <- sample(1:length(y))

# Note that I could have used sample(1:998) since I know the data contains 998 observations,
# but this way is more general.  Now let's make a new shuffled dataset with:
df.shuf <- df[CVind,]

# Note that df.shuf and df contain the exact same points, but df.shuf is in a random
# order.  Since we simulated data at random in this example, this really wasn't necessary
# but in practice, there can be correlation between observations collected close together
# so it's good practice to always shuffle the data before dividing it into groups to do
# CV.  Now I need to actually create the different groups.  First we can see how many 
# points will be left over if we simply divide by 10:
length(y) %% 10

# Now let's see how big the smallest groups will be:
length(y) %/% 10

# (Note that %/% and %% are equivalent to 'DIV' and 'MOD' if you are familar with these.)
# We see that we will have 8 observations left over, but I want to spread these extra 8
# observations across the groups.  Thus, we will want 2 groups of 99 observations and 8
# groups of 100.  Actually splitting the data into groups is a little tricky, but the 
# most explicit way to do this is simply to make a starting index (ind1) and ending index
# (ind2) for each group:

ind1 <- c(1, 101, 201, 301, 401, 501, 601, 701, 801, 900)
ind2 <- c(100, 200, 300, 400, 500, 600, 700, 800, 899, 998)

# Now we'll make a loop and each time through, we'll build one LM with the squared term
# and one without.  Each time, we'll record the test error on the corresponding hold-out
# group and save these in a vector.  Note that when you're using loops in R, it's best
# to save results to a vector that is already created to be the length you neeed -- 
# this is called preallocating.

mse.NOsq <- rep(0,10)
mse.sq <- rep(0,10)
for (i in 1:10) {
	# First we need to create the train and test data for this iteration:
	# Here we'll pull out the data from the training set
	temp.train <- df.shuf[-(ind1[i]:ind2[i]),]
	
	# And here we'll make the test set:
	temp.test.x <- df.shuf[ind1[i]:ind2[i],2:3]
	temp.test.y <- df.shuf[ind1[i]:ind2[i],1]
	
	# Now we'll build the linear models:
	lm.NOsq <- lm(y~x1,data=temp.train)
	lm.sq <- lm(y~.,data=temp.train)
	
	# And calculate the MSE on our hold-out (test) set from each:
	mse.NOsq[i] <- mean( (predict(lm.NOsq,temp.test.x) - temp.test.y)^2 )
	mse.sq[i] <- mean( (predict(lm.sq,temp.test.x) - temp.test.y)^2 )
}

# Now the MSE from each CV run is stored in our vectors; let's take the average and 
# see which model -- with or without the squared term -- did better:
mean(mse.NOsq)
mean(mse.sq)

# You should see that the average test error on the model that included the squared term
# is lower.  Finally, let's plot the results:
plot(c(1,2),c(mean(mse.NOsq),mean(mse.sq)),ylim=c(0.065,0.075),xlab="Degree of Polynomial",ylab="10-Fold CV Error")




####################################################################################
# Bootstrap & Resampling
####################################################################################
# Now let's look at bootstrapping and for this we'll do a very simple example.  Let's
# say that we want to know what the distribution of the mean looks like but we don't 
# know the central limit theorem.  Let's start with a sample of size 30 from a Poisson
# Distribution with parameter lambda = 5
x <- rpois(30,lambda=5)

# We see that the mean of our data (x) is:
mean0 <- mean(x)

# Now let's take a bunch of bootstrap samples and look at the distribution of bootstrap
# means.  Note that when we resample with the sample function in R, we want to make sure
# to do so with replacement:

nBoot <- 1000
mean.boot <- rep(0,nBoot)
for (i in 1:nBoot) {
	xperm <- sample(x,replace=T)
	mean.boot[i] <- mean(xperm)
}

# Now let's plot a histogram of the means and put a blue vertical line at where our 
# original mean was:
hist(mean.boot,freq=FALSE,breaks=20)
abline(v=mean0,col='blue',lwd=2)

# The true mean of a poisson random variable is lambda (= 5 in this case) and the true variance
# is also lambda.  By the CLT, the sampling distribution of this mean should be normal with mean
# lambda and variance lambda/n.  Let's see how well it lines up to our bootstrap distribution:
x.points <- seq(3,7,0.01)
y.points <- dnorm(x.points,mean=5,sd=sqrt(5/30))
lines(x.points, y.points, type="l",col="red",lwd=2)

# It should look pretty good, but there's a chance it may not.  The reason is that in the 
# second line (starting with 'y.points') we are using the true mean (lambda = 5) as our true
# mean value (which it is).  In practice though, we wouldn't know the value of the true mean
# (otherwise we wouldn't be trying to estimate it!) so we would instead use the value of our
# sample mean (in this case, we called it mean0).  The below code should now give you a plot 
# that looks more correctly centered. 
y.points <- dnorm(x.points,mean=mean0,sd=sqrt(5/30))
lines(x.points, y.points, type="l",col="green",lwd=2)

# Beware though:  the green density looks much more like the bootstrap histogram, but the 
# red density is the truth according to the CLT.  


####################################################################################
# Permutation Tests
####################################################################################
# Finally we look at permutation tests.  Remember that the big idea here is that under the
# null hypothesis, there is some relationship in the data (i.e. something that is implied).
# We will test this assumption by permuting some (or all) of the data and seeing whether
# the original statistic we calculated looks like it could have plausibly come from this 
# distribution.  If not, we can reject the null hypothesis.

# Here we'll look at permutation two sample (non-paired) t-test.  We have two groups
# of responses and we are interested in whether the means of the groups are the
# same:  H0:  mu_1 = mu_2   H1:  mu_1 != mu_2.  If the samples from each group were drawn 
# from normal populations with equal variances then we can use a two-sample t-test.  However,
# if this is not known a priori and/or there are not enough samples to reasonably check
# for normality and depend on the CLT, then we need another approach and permutation tests 
# are one such alternative.  Given the sample, each element in the sample can be marked/labeled 
# as being in group 1 or group 2.  Under the null hypothesis, the group means are equal and thus
# the group assignments should be arbitrary (since the null hypothesis implies that samples in the
# two groups were coming from the same population).  We can randomly permute the group labels, 
# calculate the permuted statistic, and see how far our original statistic lies from the distribution 
# of permuted statistics.  Here we'll use the sleep data and we'll use the standard t-test statistic 
# as our statistic of interest.

# First we load the data:
data()
?sleep
data(sleep)

# Now we create two groups according to the group labels in the dataset and calculate a t-statistic
# on the original group labels
Group1 <- sleep$extra[sleep$group==1]
Group2 <- sleep$extra[sleep$group==2]
t0 <- abs(t.test(Group1,Group2)$statistic)

# Now we'll do 1000 permutations of the group labels and calculate a new t-statistic each time:
nperm <- 1000
t.perm <- rep(0,nperm)
for (i in 1:nperm) {
	ind <- sample(sleep$group)
	sleep$group <- ind
	Group1 <- sleep$extra[sleep$group==1]	
	Group2 <- sleep$extra[sleep$group==2]	
	t.perm[i] <- abs(t.test(Group1,Group2)$statistic)
}

# Now let's make a histogram of our permuted t-statistics and see where
# our original t-statistic falls:
hist(t.perm)
abline(v=t0,col='blue',lwd=2)

# You should see that it looks like it could have plausibly come from this distribution, so we
# probably not reject the null hypothesis.  Let's be sure by calculating an explicit p-value.  
# Here, the p-value is just the percentage of permutation statistics that fell above (were more
# extreme than) our original t-statistic.
p <- mean(t.perm > t0)

# You should see that the p-value is larger than 0.05, so we cannot reject the null hypothesis that
# the group means are equal