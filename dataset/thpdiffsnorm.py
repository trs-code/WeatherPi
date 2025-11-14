import pandas as pd

thpdf = pd.read_csv("THPDiffs.csv")

tempCol = thpdf[["Temperature"]]
humCol = thpdf[["Humidity"]]
pressCol = thpdf[["Pressure"]]
temp1Col = thpdf[["1HRTD"]]
hum1Col = thpdf[["1HRHD"]]
press1Col = thpdf[["1HRPD"]]
temp6Col = thpdf[["6HRTD"]]
hum6Col = thpdf[["6HRHD"]]
press6Col = thpdf[["6HRPD"]]
precCol = thpdf[["Precipitation"]]

def colNorm(col):
    minVal = col.min()
    maxVal = col.max()
    return (col-minVal)/(maxVal-minVal)

tempColNorm = colNorm(tempCol)
humColNorm = colNorm(humCol)
pressColNorm = colNorm(pressCol)
temp1ColNorm = colNorm(temp1Col)
hum1ColNorm = colNorm(hum1Col)
press1ColNorm = colNorm(press1Col)
temp6ColNorm = colNorm(temp6Col)
hum6ColNorm = colNorm(hum6Col)
press6ColNorm = colNorm(press6Col)

combCols = pd.concat([tempColNorm, humColNorm, pressColNorm, temp1ColNorm, hum1ColNorm, press1ColNorm, temp6ColNorm, hum6ColNorm, press6ColNorm, precCol], axis = 1)
combCols.to_csv("THPDiffsNorm.csv", index = False)

