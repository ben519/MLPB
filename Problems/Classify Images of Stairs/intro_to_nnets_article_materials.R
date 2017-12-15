# Datasets and graphs for the article Introduction To Neural Networks
# https://gormanalysis.com/introduction-to-neural-networks

#======================================================================================================
# settings

options(scipen = 20)

#--------------------------------------------------
# packages

library(data.table)
library(ggplot2)
library(stringr)

#======================================================================================================
# Helper methods

logistic1 <- function(x = seq(-10, 10, length.out = 1000), a = 1, slope = 1, offset = 0){
  a / (1 + exp(-slope * (x - offset)))
}

sigmoid <- function(x){
  logistic1(x, a = 1, slope = 1, offset = 0)
}

#======================================================================================================
# Datasets

# Read train & test data
train <- fread("https://raw.githubusercontent.com/ben519/MLPB/master/Problems/Classify%20Images%20of%20Stairs/_Data/train.csv")
test <- fread("https://raw.githubusercontent.com/ben519/MLPB/master/Problems/Classify%20Images%20of%20Stairs/_Data/test.csv")

# Insert Label for plots
train[, Label := ifelse(IsStairs, "Stairs", "Not Stairs")]
train[, Label := factor(Label, levels=c("Not Stairs", "Stairs"))]
test[, Label := ifelse(IsStairs, "Stairs", "Not Stairs")]
test[, Label := factor(Label, levels=c("Not Stairs", "Stairs"))]

# Reshape from wide to tall
trainTall <- melt(
  train, 
  id.vars = c("ImageId", "IsStairs", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"), 
  value.name = "Intensity",
  variable.name = "Pixel"
)

testTall <- melt(
  test, 
  id.vars = c("ImageId", "IsStairs", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"), 
  value.name = "Intensity",
  variable.name = "Pixel"
)

trainTall[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
testTall[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]

#======================================================================================================
# Figures

#--------------------------------------------------
# fig1: stairs vs not stairs

fig1data <- trainTall[ImageId %in% train$ImageId[1:12]]
fig1 <- ggplot(fig1data, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")+
  facet_wrap(~ImageId, nrow = 2, labeller = labeller(ImageId = setNames(as.character(fig1data$Label), fig1data$ImageId)))

#--------------------------------------------------
# fig2: image with pixels labeled x1, x2, x3, x4

image1 <- trainTall[ImageId == 1]
image1[Row == 1 & Col == 1, PixelLabel := paste0("x1 = ", Intensity)]
image1[Row == 1 & Col == 2, PixelLabel := paste0("x2 = ", Intensity)]
image1[Row == 2 & Col == 1, PixelLabel := paste0("x3 = ", Intensity)]
image1[Row == 2 & Col == 2, PixelLabel := paste0("x4 = ", Intensity)]

fig2lables <- image1
fig2lables[, `:=`(x = Col+0.45, y = 1.55 - Row)]

fig2 <- ggplot(image1, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  geom_text(data = fig2lables, aes(x = x, y = y, label = PixelLabel), color = "goldenrod3", size = 5, hjust = 1, vjust = 0)+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")+
  labs(title = image1$Label[1])+
  theme(plot.title = element_text(hjust = 0.5, vjust = -50))

#======================================================================================================
# Perceptron model

#--------------------------------------------------
# fig3: pictures of stairs and non-stairs with yhat & prediction
# Build a grid with columns (Stairs, NotStairs) and rows (Predicted Stairs, Predicted Not Stairs)

preds3 <- train[ImageId %in% c(4, 240, 257, 367, 407, 495)]
preds3[, yhat := -0.0019*R1C1 + -0.0016*R1C2 + 0.0020*R2C1 + 0.0023*R2C2 + 0.0003]
preds3[, PredStairs := yhat > 0]
preds3[, Label := paste0(
  "x = [", R1C1, ", ", R1C2, ", ", R2C1, ", ", R2C2, "]",
  "\nŷ = ", format(round(yhat, 2), nsmall = 2), " → ", ifelse(yhat > 0, "Stairs", "Not Stairs")
)]
preds3[, Label := factor(Label, unique(Label[order(yhat)]))]
fig3_data <- melt(
  data = preds3, 
  id.vars = c("ImageId", "yhat", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"),
  variable.name = "Pixel",
  value.name = "Intensity"
)
fig3_data[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
fig3 <- ggplot(fig3_data, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  facet_wrap(~Label, nrow=1, labeller = labeller(ImageId = setNames(preds3$Label, preds3$ImageId)))+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")

#--------------------------------------------------
# fig4 & fig5: Cases A and B to show problem with linear relationship

cases <- data.table(
  Case = c("A", "A", "B", "B"),
  Label = factor(c("Before", "After", "Before", "After"), levels=c("Before", "After")), 
  R1C1=c(100, 100, 100, 100), R1C2=c(0, 0, 0, 0), R2C1=c(0, 60, 60, 120), R2C2=c(125, 125, 125, 125)
)
cases[, yhat := -0.0019*R1C1 + -0.0016*R1C2 + 0.0020*R2C1 + 0.0023*R2C2 + 0.0003]
cases[, Label := paste0(
  "x = [", R1C1, ", ", R1C2, ", ", R2C1, ", ", R2C2, "]",
  "\nŷ = ", format(round(yhat, 2), nsmall = 2)
)]
cases[, Label := factor(Label, unique(Label[order(yhat)]))]
casesTall <- melt(
  cases, 
  id.vars = c("Case", "Label", "yhat"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"), 
  value.name = "Intensity",
  variable.name = "Pixel"
)
casesTall[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
fig4 <- ggplot(casesTall[Case == "A"], aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")+facet_wrap(~Label, nrow=1)
fig5 <- ggplot(casesTall[Case == "B"], aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")+facet_wrap(~Label, nrow=1)

#======================================================================================================
# Linear Perceptron with Sigmoid Activation

#--------------------------------------------------
# fig6: pictures of stairs and non-stairs with yhat & prediction
# Build a grid with columns (Stairs, NotStairs) and rows (Predicted Stairs, Predicted Not Stairs)

preds6 <- train[ImageId %in% c(4, 240, 257, 367, 407, 495)]
preds6[, z := -0.196*R1C1 + -0.189*R1C2 + 0.136*R2C1 + 0.109*R2C2 + -0.006]
preds6[, yhat := sigmoid(z)]
preds6[, PredStairs := yhat > 0.5]
preds6[, Label := paste0(
  "x = [", R1C1, ", ", R1C2, ", ", R2C1, ", ", R2C2, "]",
  "\nw·x+b = ", format(round(z, 1), nsmall = 1),
  "\nŷ = ", format(round(yhat, 2), nsmall = 2), " → ", ifelse(yhat > 0.5, "Stairs", "Not Stairs")
)]
preds6[, Label := factor(Label, unique(Label[order(yhat)]))]
fig6_data <- melt(
  data = preds6, 
  id.vars = c("ImageId", "yhat", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"),
  variable.name = "Pixel",
  value.name = "Intensity"
)
fig6_data[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
fig6 <- ggplot(fig6_data, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  facet_wrap(~Label, nrow=1, labeller = labeller(ImageId = setNames(preds6$Label, preds6$ImageId)))+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")

#--------------------------------------------------
# fig7 & fig8: Cases A and B to show problem with linear relationship

cases <- data.table(
  Case = c("A", "A", "B", "B"),
  Label = factor(c("Before", "After", "Before", "After"), levels=c("Before", "After")), 
  R1C1=c(100, 100, 100, 100), R1C2=c(0, 0, 0, 0), R2C1=c(0, 60, 60, 120), R2C2=c(125, 125, 125, 125)
)
cases[, z := -0.196*R1C1 + -0.189*R1C2 + 0.136*R2C1 + 0.109*R2C2 + -0.006]
cases[, yhat := round(sigmoid(-0.196*R1C1 + -0.189*R1C2 + 0.136*R2C1 + 0.109*R2C2 + -0.006), 3)]
cases[, Label := paste0(
  "x = [", R1C1, ", ", R1C2, ", ", R2C1, ", ", R2C2, "]",
  "\nŷ = ", format(round(yhat, 2), nsmall = 2)
)]
cases[, Label := factor(Label, unique(Label[order(yhat)]))]
casesTall <- melt(
  cases, 
  id.vars = c("Case", "Label", "yhat"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"), 
  value.name = "Intensity",
  variable.name = "Pixel"
)
casesTall[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
fig7 <- ggplot(casesTall[Case == "A"], aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")+facet_wrap(~Label, nrow=1)
fig8 <- ggplot(casesTall[Case == "B"], aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")+facet_wrap(~Label, nrow=1)

#--------------------------------------------------
# fig9 | Case A and B difference on sigmoid curve

dt <- data.table(x = seq(-12, 12, length.out = 1000))
dt[, y := sigmoid(x)]
cases[, plotx := z + c(0, -.05, 0.05, 0)]
cases[, Label2 := c("A: Before", "A: After", "B: Before", "B: After")]
cases[, `:=`(LabelX = plotx + c(-1.1, -1.5, 1.5, 1.1), LabelY = yhat + c(0.03, 0, 0, -.03), Angle = c(0, 0, 0, 0))]

fig9 <- ggplot(dt, aes(x=x, y=y))+geom_line()+
  geom_point(data = cases, aes(x=plotx, y=yhat, color=Case))+
  geom_vline(data = cases, aes(xintercept=plotx, color=Case), linetype = "dashed", size=0.75)+
  geom_text(data = cases, aes(label = Label2, x = LabelX, y = LabelY, angle = Angle))+
  labs(x = "z = w · x", y = "ŷ = sigmoid(z)")+
  guides(color=FALSE)

#======================================================================================================
# 2-layer perceptron with sigmoid activation

#--------------------------------------------------
# fig10 | combine left stairs model and right stairs model to predict stairs

preds10 <- copy(train[ImageId %in% c(4, 240, 257, 367, 407, 495)])
preds10[, `:=`(
  z_LeftStairs = 0.002*R1C1 + -0.05*R1C2 + 0.012*R2C1 + 0.012*R2C2 - .05,
  z_RightStairs = -0.05*R1C1 + 0.002*R1C2 + 0.012*R2C1 + 0.012*R2C2 - .05
)]
preds10[, `:=`(
  yhat_LeftStairs = sigmoid(z_LeftStairs),
  yhat_RightStairs = sigmoid(z_RightStairs)
)]
preds10[, z := 3*yhat_LeftStairs + 3*yhat_RightStairs - 1]
preds10[, yhat := sigmoid(z)]
preds10[, PredStairs := yhat > 0.5]
preds10[, Label := paste0(
  "x = [", R1C1, ", ", R1C2, ", ", R2C1, ", ", R2C2, "]",
  "\nŷ_left = ", format(round(yhat_LeftStairs, 2), nsmall = 2), ", ŷ_right = ", format(round(yhat_RightStairs, 2), nsmall = 2),
  "\nŷ = ", format(round(yhat, 2), nsmall = 2), " → ", ifelse(yhat > 0.5, "Stairs", "Not Stairs")
)]
preds10[, Label := factor(Label, unique(Label[order(yhat)]))]
fig10_data <- melt(
  data = preds10, 
  id.vars = c("ImageId", "yhat", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"),
  variable.name = "Pixel",
  value.name = "Intensity"
)
fig10_data[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
fig10 <- ggplot(fig10_data, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  facet_wrap(~Label, nrow=1, labeller = labeller(ImageId = setNames(preds10$Label, preds10$ImageId)))+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")

#--------------------------------------------------
# fig11 | combine dark bottom row model, (dark x1 and light x2) and (light x1 and dark x2)

preds11 <- copy(train[ImageId %in% c(4, 240, 257, 367, 407, 495)])
preds11[, `:=`(
  z_DarkBottom = 0*R1C1 + 0*R1C2 + .2*R2C1 + .2*R2C2 - .5,
  z_DarkX1 = 0.1*R1C1 + -0.1*R1C2 + 0*R2C1 + 0*R2C2 - .05,
  z_DarkX2 = -0.1*R1C1 + 0.01*R1C2 + 0*R2C1 + 0*R2C2 - .05
)]
preds11[, `:=`(
  yhat_DarkBottom = sigmoid(z_DarkBottom),
  yhat_DarkX1 = sigmoid(z_DarkX1),
  yhat_DarkX2 = sigmoid(z_DarkX2)
)]
preds11[, z := 3*yhat_DarkBottom + 3*yhat_DarkX1 + 3*yhat_DarkX2 - 4]
preds11[, yhat := sigmoid(z)]
preds11[, PredStairs := yhat > 0.5]
preds11[, Label := paste0(
  "x = [", R1C1, ", ", R1C2, ", ", R2C1, ", ", R2C2, "]",
  "\nŷ1 = ", format(round(yhat_DarkBottom, 1), nsmall = 1), ", ŷ2 = ", format(round(yhat_DarkX1, 1), nsmall = 1), ", ŷ3 = ", format(round(yhat_DarkX2, 1), nsmall = 1),
  "\nŷ = ", format(round(yhat, 2), nsmall = 2), " → ", ifelse(yhat > 0.5, "Stairs", "Not Stairs")
)]
preds11[, Label := factor(Label, unique(Label[order(yhat)]))]
fig11_data <- melt(
  data = preds11, 
  id.vars = c("ImageId", "yhat", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"),
  variable.name = "Pixel",
  value.name = "Intensity"
)
fig11_data[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
fig11 <- ggplot(fig11_data, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  facet_wrap(~Label, nrow=1, labeller = labeller(ImageId = setNames(preds11$Label, preds11$ImageId)))+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")

#--------------------------------------------------
# fig12 | predict light stairs

preds12 <- train[ImageId %in% c(1, 2, 3, 4, 5, 6)]
preds12[, `:=`(R1C1=c(15,5,50,250,0,0), R1C2=c(10,30,0,0,20,100), R2C1=c(5,30,50,250,20,75), R2C2=c(20,30,50,250,0,125))]
preds12[, IsStairs := c(F, T, T, T, F, T)]

preds12[, z_shadedbtm := 0*R1C1 + 0*R1C2 + .06*R2C1 + .06*R2C2 - 3]
preds12[, z_shadedX1 := 0.06*R1C1 + -0.4*R1C2 + 0*R2C1 + 0*R2C2 - 1.5]
preds12[, z_shadedX2 := -0.4*R1C1 + 0.06*R1C2 + 0*R2C1 + 0*R2C2 - 1.5]

preds12[, z_darkbtm := 0*R1C1 + 0*R1C2 + .06*R2C1 + .06*R2C2 - 10]
preds12[, z_darkX1 := 0.06*R1C1 + -0.4*R1C2 + 0*R2C1 + 0*R2C2 - 3.5]
preds12[, z_darkX2 := -0.4*R1C1 + 0.06*R1C2 + 0*R2C1 + 0*R2C2 - 3.5]

preds12[, `:=`(
  yhat_shadedbtm = sigmoid(z_shadedbtm),
  yhat_shadedX1 = sigmoid(z_shadedX1),
  yhat_shadedX2 = sigmoid(z_shadedX2),
  
  yhat_darkbtm = sigmoid(z_darkbtm),
  yhat_darkX1 = sigmoid(z_darkX1),
  yhat_darkX2 = sigmoid(z_darkX2)
)]
preds12[, z := 8*(yhat_shadedbtm - yhat_darkbtm) + (yhat_shadedX1 - yhat_darkX1) + (yhat_shadedX2 - yhat_darkX2) - 2]
preds12[, yhat := sigmoid(z)]
preds12[, PredStairs := yhat > 0.5]
preds12[, Label := paste0(
  "x = [", R1C1, ", ", R1C2, ", ", R2C1, ", ", R2C2, "]",
  "\nŷ1 = ", format(round(yhat_shadedbtm, 2), nsmall = 1), ", ŷ2 = ", format(round(yhat_shadedX1, 2), nsmall = 1), ", ŷ3 = ", format(round(yhat_shadedX2, 2), nsmall = 1),
  "\nŷ4 = ", format(round(yhat_darkbtm, 2), nsmall = 1), ", ŷ5 = ", format(round(yhat_darkX1, 2), nsmall = 1), ", ŷ6 = ", format(round(yhat_darkX2, 2), nsmall = 1),
  "\nŷ = ", format(round(yhat, 2), nsmall = 2), " → ", ifelse(yhat > 0.5, "Stairs", "Not Stairs")
)]
preds12[, Label := factor(Label, unique(Label[order(yhat)]))]
fig12_data <- melt(
  data = preds12, 
  id.vars = c("ImageId", "yhat", "Label"), 
  measure.vars = c("R1C1", "R1C2", "R2C1", "R2C2"),
  variable.name = "Pixel",
  value.name = "Intensity"
)
fig12_data[, `:=`(Row = as.integer(str_extract(Pixel, "(?<=R)\\d")), Col = as.integer(str_extract(Pixel, "(?<=C)\\d")))]
fig12 <- ggplot(fig12_data, aes(x = Col, y = 2-Row, fill = Intensity))+geom_tile(color="white")+
  facet_wrap(~Label, nrow=1, labeller = labeller(ImageId = setNames(preds12$Label, preds12$ImageId)))+
  scale_fill_gradient(low = "white", high = "black", limits=c(0, 255), guide=FALSE)+
  theme_void()+labs(x="", y="")
