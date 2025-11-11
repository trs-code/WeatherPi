import pandas as pd

thf = pd.read_csv("TH.csv")
pf = pd.read_csv("P.csv")
onehrf = pd.read_csv("1HR.csv")

tempCol = thf[["Temperature"]]
humCol = thf[["Humidity"]]
pressCol = (pf[["Pressure"]] - 0.17).round(2)
precCol = onehrf[["Precipitation"]]

combCols = pd.concat([tempCol, humCol,pressCol, precCol], axis = 1)
combCols.to_csv("THP.csv", index = False)
#thpdf = "THPDiffs.csv"

