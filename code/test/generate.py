# run the program with command parameter
#    python generate.py [number of test cases]
# before running the test, please make sure ./convolution_python and ./convolution_model exist
#    make sure there are same number of test cases generated from model (convultion result of Y) in ./convolution_model   

import random
import sys
import pdb

import numpy as np
from scipy import signal
import os


inputSize = 100 #size of the input image 
filterSize = 3  #size of the filter
channelN = 3  # number of RGB channel 

prefix = "./convolution_python/"
prefixModel = "./convolution_model/"
decimal = 2
accuracy  = "%." + str(decimal) + "f "

# clear test cases in current directory and convultion directory (Y)

def clear ():
    os.system("rm -rf x*")
    os.system("rm -rf w*")
    os.system("rm -rf " + prefix + "y*")


# generate test cases:
#    X matrix of size inputSize*inputSize
#    w matrix of size filterSize*filterSize
def generateX():

    #pdb.set_trace()
    inputNum = sys.argv[1]
    
    for lo in range(0, int(inputNum)):
        for RGB in range(0, channelN):
#            fx = open("x" + str(RGB + 1) + "-" + str(lo + 1), "w")
#            fw = open("w" + str(RGB + 1) + "-" + str(lo + 1), "w")
            matX = []
            for i in range(0, inputSize):    
                l = []
                for j in range(0, inputSize):
                    e = random.random()
                    
                    l. append(e)
                matX.append(l)

            matW = []
            for i in range(0, filterSize):    
                l = []
                for j in range(0, filterSize):
                    e = random.random()    
                    l. append(e)
                matW.append(l)
            
            fx = open("x" + str(RGB + 1) + "-" + str(lo + 1), "w")
            fw = open("w" + str(RGB + 1) + "-" + str(lo + 1), "w")

            for l in matX:
                #print l
                for e in l:
                    fx.write('%.2f ' %e)
                fx.write('\n')

            for l in matW:
                #print l                
                for e in l:
                    fw.write('%.2f ' %e)
                fw.write("\n")

# generate Y from test cases 
def generateY():    
    inputNum = int(sys.argv[1])
    for lo in range(0, inputNum):
        gradlist = []
        for RGB in range(0, channelN):
            fx = open("x" + str(RGB + 1) + "-" + str(lo + 1), "r")
            fw = open("w" + str(RGB + 1) + "-" + str(lo + 1), "r")
                        
            matX = []
            matW = []
            for line in fx.readlines():
                lstr = line.split(' ')
                del lstr[-1]
                l = [float(e) for e in lstr]                
                matX.append(l)

            for line in fw.readlines():
                lstr = line.split(' ')
                del lstr[-1]
                l = [float(e) for e in lstr]
                matW.append(l)

            matWnp = np.array(matW)
            matXnp = np.array(matX)
            
#            pdb.set_trace()    
            grad = signal.convolve2d(matXnp, matWnp, boundary='symm', mode='same')
            gradlist.append(grad)

            # write to file for each channel
            fcon = open(prefix + "y" + str(RGB + 1) + "-" + str(lo + 1), "w")
            for l in grad:
                #print l                
                for e in l:
                    fcon.write(accuracy %e)
                fcon.write("\n")

        # add up channels
        matCon = []
        for i in range (0, len(grad)):
            l = []
            for j in range (0, len(grad[0])):
                x = 0.0
                
                for k in range(0, len(gradlist)):
                    x  = x + gradlist[k][i][j]
                l.append(x)
            matCon.append(l)

        # write y to file 
        fy = open(prefix + "y" + "y" + "-" + str(lo + 1), "w")
        for l in grad:
                #print l                
            for e in l:
                fcon.write(accuracy %e)
            fy.write("\n")


            
# compare Y from python to Y from model  
def test():
    # read result from python
    for lo in range(0, int(inputNum)):
        fy = open(prefix + "y" + "y" + "-" + str(lo + 1), "r")
        YfromPython = []
        for line in fy.readlines():
            l = [float(e) for e in line.split(' ')]
            del l[inputSize]
            YfromPython.append[l]

##$ name of the file needs to be changed to actual generated files from model
        fy = open(prefixModel + "y" + "y" + "-" + str(lo + 1), "r")
        YfromModel = []
        for line in fy.readlines():
            l = [float(e) for e in line.split(' ')]
            del l[inputSize]
            YfromModel.append[l]

        for i in range(0, inputSize):
            for j in range(0, inputSize):
                if round(YfromPython[i][j], decimal) != YfromModel:
                    print "not match: sample:" + str(lo)
                    print "i:" + str(i)
                    print "j" + str(j)
                    
            
if __name__ == '__main__':

    clear()
    generateX()
    generateY()
    test()
