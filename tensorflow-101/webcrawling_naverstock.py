#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import urllib
from bs4 import BeautifulSoup
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

print ("Packages loaded")
dates = []
date_strs = []
last_trade_prices = []
for nrpage in range(200):
    rawurl = "http://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page="
    url = rawurl + str(nrpage + 1)
    # Get data (parse)
    soup = BeautifulSoup(urllib.urlopen(url).read())
    dateinfo = soup.find_all('td', {'class': 'date'})
    valueinfo = soup.find_all('td', {'class': 'number_1'})
    nrdata = len(valueinfo)
    for i in range(nrdata):
        # Date
        currdate = str(dateinfo[int(i / 4)])
        currdate = currdate.replace('<td class="date">', '')
        currdate = currdate.replace('</td>', '')
        currdate = currdate.replace('.', "")

        # Values
        currdata = str(valueinfo[i])
        currdata = currdata.replace('<td class="number_1">', "")
        currdata = currdata.replace('</td>', "")
        currdata = currdata.replace('<span class="tah p11 red01">', '')
        currdata = currdata.replace('</span>', "")
        currdata = currdata.replace('<td class="number_1" style="padding-right:40px;">', "")
        currdata = currdata.replace('<td class="number_1" style="padding-right:30px;">', "")
        currdata = currdata.replace('<span class="tah p11 nv01">', "")
        currdata = currdata.replace(' ', "")
        currdata = currdata.replace('\n', "")
        currdata = currdata.replace('\t', "")
        currdata = currdata.replace(',', "")

        if i % 4 == 0:
            print ("\nCurr date is %s" % (currdate))
            date_strs.append(currdate)
            dates.append(float(currdate))
            print ("Last traded price: %s" % (currdata))
            last_trade_prices.append(float(currdata))
        elif i % 4 == 1:
            print ("Fluctuation ratio: %s" % (currdata))
        elif i % 4 == 2:
            print ("Traded volume:     %s" % (currdata))
        elif i % 4 == 3:
            print ("Traded price:      %s" % (currdata))
# Reverse
dates.reverse()
date_strs.reverse()
last_trade_prices.reverse()

plt.figure(1, figsize=(14, 7))
n = len(last_trade_prices)
x = range(n)
step = int(n / 20)
xtick_input = x[0:n:step]
xtick_input.append(x[-1])
xtick_str = date_strs[0:n:step]
xtick_str.append(date_strs[-1])
plt.xticks(xtick_input, xtick_str, fontsize=20
           , rotation='vertical')
plt.plot(x, last_trade_prices, "-")
plt.title("KOSPI")


# GPR on this data
def kernel_se(X1, X2, g2, l2, w2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            x1 = X1[i, :]
            x2 = X2[j, :]
            d = x1 - x2
            K[i, j] = g2 * np.exp(-d * d / (l2))
            if n1 > 1 and i == j:
                K[i, j] = K[i, j] + w2
    return K


print ("Kernel function defined")

# gpr
xdata = np.array(x).reshape(-1, len(x)).T
ydata = np.array(last_trade_prices).reshape(-1, len(last_trade_prices)).T
ydata_mz = ydata - np.mean(ydata)
xtest = xtest = np.array([np.linspace(0, 1200, 3000)]).T
g2 = 1
l2 = 30
w2 = 1e-1
Kdata = kernel_se(xdata, xdata, g2, l2, w2)
alpha = np.matmul(np.linalg.inv(Kdata), ydata_mz)
Ktest = kernel_se(xtest, xdata, g2, l2, w2)
ytest = np.matmul(Ktest, alpha) + np.mean(ydata)
print ("GPR ready")

# Plot
plt.figure(1, figsize=(15, 8))
plt.plot(xdata[:, 0], ydata[:, 0], 'r-', label='Original data')
plt.plot(xtest[:, 0], ytest[:, 0], 'b-', label='GPR')
plt.legend()
