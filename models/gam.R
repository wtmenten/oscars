install.packages('gam')
install.packages('gamlss')
library('gam')
library('gamlss')
require('gam')
getwd()
# use csv file in root dir of github
oscars <- read.table(file.choose(), header=T, sep=",") #choose AA_4_computed
summary(oscars)
str(oscars[0])

# convert T/F vals to binary
oscars[3] = sapply(oscars[3], as.numeric) -1
# convert string nums to num
oscars[, c(8:11)] <- sapply(sapply(oscars[, c(8:11)], as.character), as.numeric)
oscars[, c(6:11)] <- scale(oscars[, c(6:11)])
oscars[, c(18)] <- scale(oscars[, c(18)])
# split the data
oscars
splitindex = floor(nrow(oscars) * .1)
test = oscars[1:splitindex,]
train = oscars [splitindex:nrow(oscars),]
tl = subset(test, Winner == 0)
tw = subset(test, Winner == 1)

reportgam <- function(gamobj) {
  gamobj$xlevels[["Producer"]] <- levels(oscars$Producer)
  wpreds = predict.gam(gamobj,tw)
  lpreds = predict.gam(gamobj,tl)
  plot(wpreds, col='green')
  points(lpreds, col='red')
  print(summary(gamobj))
}

 # minimum first
gamobj<-gam(Winner ~ Producer + log(IMDB.Votes+1) + IMDB.Rating,family=binomial,data=train)
reportgam(gamobj)



# with more ratings
gamobj<-gam(Winner ~ Producer + log(IMDB.Votes+1) + IMDB.Rating + Average.Critic.Score + Average.Audience.Score,family=binomial,data=train)
reportgam(gamobj)


# with norm budget and gross
gamobj<-gam(Winner ~ Producer + log(IMDB.Votes+1) + IMDB.Rating + Average.Critic.Score + Average.Audience.Score  + Normalized.Gross + Normalized.Budget,family=binomial,data=train)
reportgam(gamobj)


# with ROI
gamobj<-gam(Winner ~ Producer + log(IMDB.Votes+1) + IMDB.Rating + Average.Critic.Score + Average.Audience.Score  + Normalized.Gross + Normalized.Budget + Return.on.Investment,family=binomial,data=train)
reportgam(gamobj)


# gamobj<-gam(Winner ~ factor(Producer) + IMDB.Votes + IMDB.Rating + Return.on.Investment + Average.Critic.Score + Average.Audience.Score + Normalized.Budget + Normalized.Gross,family=binomial,data=train)

