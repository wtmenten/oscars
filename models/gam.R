install.packages('gam')
install.packages('gamlss')
library('gam')
library('gamlss')

getwd()
# use csv file in root dir of github
oscars <- read.table(file.choose(), header=T, sep=",") #choose AA_4_computed_new
summary(oscars)

# convert T/F vals to binary
oscars[3] = sapply(oscars[3], as.numeric) -1

# convert string cols to num
oscars[, c(8:11)] <- sapply(sapply(oscars[, c(8:11)], as.character), as.numeric)

# standard scale numeric columns
oscars[, c(6:11)] <- scale(oscars[, c(6:11)])
oscars[, c(18)] <- scale(oscars[, c(18)])

# split the data
splitindex = floor(nrow(oscars) * .1)
test = oscars[1:splitindex,]
train = oscars [splitindex:nrow(oscars),]

# testing data split by winners and losers for graph color-coding
tl = subset(test, Winner == 0)
tw = subset(test, Winner == 1)

reportgam <- function(gamobj) {
  # from when we used producer factor. This ensured that all producers are included as possible levels
  # gamobj$xlevels[["Producer"]] <- levels(oscars$Producer)
  wpreds = predict.gam(gamobj,tw) # predictions for winners set
  lpreds = predict.gam(gamobj,tl) # predictions for losers set
  plot(wpreds, col='green')
  points(lpreds, col='red')
  print(summary(gamobj))
  print(gamobj$coefficients)
}

# matching the human study first
gamobj<-gam(Winner ~ + log(IMDB.Votes+1) + IMDB.Rating +  Average.Critic.Score + Average.Audience.Score  + Normalized.Gross + Normalized.Budget + Return.on.Investment,family=binomial,data=train)
reportgam(gamobj)
