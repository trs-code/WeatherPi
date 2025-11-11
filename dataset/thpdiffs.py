import pandas as pd

thpf = pd.read_csv("THP.csv")

tempCol = thpf[["Temperature"]]
humCol = thpf[["Humidity"]]
pressCol = thpf[["Pressure"]]
thpf[["1HRTD"]] = tempCol - tempCol.shift(periods=1)
thpf[["1HRHD"]] = humCol - humCol.shift(periods=1)
thpf[["1HRPD"]] = pressCol - pressCol.shift(periods=1)
thpf[["6HRTD"]] = (tempCol - tempCol.shift(periods=6))/6
thpf[["6HRHD"]] = (humCol - humCol.shift(periods=6))/6
thpf[["6HRPD"]] = (pressCol - pressCol.shift(periods=6))/6

onehrtemp = thpf[["1HRTD"]].round(2)
onehrhum = thpf[["1HRHD"]].round(2)
onehrpress = thpf[["1HRPD"]].round(2)
sixhrtemp = thpf[["6HRTD"]].round(2)
sixhrhum = thpf[["6HRHD"]].round(2)
sixhrpress = thpf[["6HRPD"]].round(2)

precCol = thpf[["Precipitation"]]

combCols = pd.concat([tempCol, humCol, pressCol, onehrtemp, onehrhum, onehrpress, sixhrtemp, sixhrhum, sixhrpress,precCol], axis = 1)
combCols.to_csv("THPDiffs.csv", index = False)
#thpdf = "THPDiffs.csv"

