#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np

print ("Loading package(s)")
print ("Hello, world")

# THERE ARE THREE POPULAR TYPES
# 1. INTEGER
x = 3;
print ("Integer: %01d, %02d, %03d, %04d, %05d" % (x, x, x, x, x))
# 2. FLOAT
x = 123.456;
print ("Float: %.0f, %.1f, %.2f, %1.2f, %2.2f" % (x, x, x, x, x))

# 3. STRING
x = "Hello, world"
print ("String: [%s], [%3s], [%20s]" % (x, x, x))

dlmethods = ["ANN", "MLP", "CNN", "RNN", "DAE"]

for alg in dlmethods:
    if alg in ["ANN", "MLP"]:
        print ("We have seen %s" % (alg))

for alg in dlmethods:
    if alg in ["ANN", "MLP", "CNN"]:
        print ("%s is a feed-forward network." % (alg))
    elif alg in ["RNN"]:
        print ("%s is a recurrent network." % (alg))
    else:
        print ("%s is an unsupervised method." % (alg))

# Little more advanced?
print("\nFOR loop with index.")
for alg, i in zip(dlmethods, range(len(dlmethods))):
    if alg in ["ANN", "MLP", "CNN"]:
        print ("[%d/%d] %s is a feed-forward network."
               % (i, len(dlmethods), alg))
    elif alg in ["RNN"]:
        print ("[%d/%d] %s is a recurrent network."
               % (i, len(dlmethods), alg))
    else:
        print ("[%d/%d] %s is an unsupervised method."
               % (i, len(dlmethods), alg))

# Function definition looks like this
def sum(a, b):
    return a+b
X = 10.
Y = 20.
# Usage
print ("%.1f + %.1f = %.1f" % (X, Y, sum(X, Y)))

head = "Deep learning"
body = "very "
tail = "HARD."
print (head + " is " + body + tail)

# Repeat words
print (head + " is " + body*3 + tail)
print (head + " is " + body*10 + tail)

# It is used in this way
print ("\n" + "="*50)
print (" "*15 + "It is used in this way")
print ("="*50 + "\n")

# Indexing characters in the string
x = "Hello, world"
for i in range(len(x)):
    print ("Index: [%02d/%02d] Char: %s"
           % (i, len(x), x[i]))

# More indexing
idx = -2
print ("(%d)th char is %s" % (idx, x[idx]))
idxfr = 0
idxto = 8
print ("String from %d to %d is [%s]" % (idxfr, idxto, x[idxfr:idxto]))
idxfr = 4
print ("String from %d to END is [%s]" % (idxfr, x[idxfr:]))
x = "20160607Cloudy"
year = x[:4]
day = x[4:8]
weather = x[8:]
print ("[%s] -> [%s] + [%s] + [%s] " % (x, year, day, weather))

#list
a = []
b = [1, 2, 3]
c = ["Hello", ",", "world"]
d = [1, 2, 3, "x", "y", "z"]
x = []
print x
x.append('a')
print x
x.append(123)
print x
x.append(["a", "b"])
print x
print ("Length of x is %d " % (len(x)))
for i in range(len(x)):
    print ("[%02d/%02d] %s" % (i, len(x), x[i]))


#dict
dic = dict()
dic["name"] = "Sungjoon"
dic["age"] = 31
dic["job"] = "Ph.D. Candidate"

print dic


#class
class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print ('HELLO, %s!'
                   % self.name.upper())
        else:
            print ('Hello, %s'
                   % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"

def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))
    print

#array
x = np.array([1, 2, 3]) # rank 1 array
print_np(x)

x[0] = 5
print_np(x)

y = np.array([[1,2,3], [4,5,6]])
print_np(y)

a = np.zeros((3, 2))
print_np(a)

b = np.ones((1, 2))
print_np(b)

c = np.eye(2, 2)
print_np(c)

d = np.random.random((2, 2))
print_np(d)

e = np.random.randn(1, 10)
print_np(e)


# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print_np(a)

# Use slicing to pull out the subarray consisting
# of the first 2 rows
# and columns 1 and 2; b is the following array
# of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print_np(b)

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print_np(a)

row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a

print_np(row_r1)
print_np(row_r2)
print_np(row_r3)


x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # particular datatype

print_np(x)
print_np(y)
print_np(z)

#array math
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print (x + y)
print (np.add(x, y))

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)    # Create an empty matrix
                        # with the same shape as x

print_np(x)
print_np(v)
print_np(y)