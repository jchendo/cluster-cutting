import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.spatial.distance import cdist

NLXHeaderSize = 16384
TETRODERECORDSIZE = 304
RecordOffset = 48
fileName = "data/8360_S32_Sc2/8360_S32_Sc2.Ntt"

insweeps = np.zeros(128, dtype=np.int16)
separateSweeps = np.zeros((4,32),dtype=np.int16)
with open(fileName,'rb') as datafile:
    fd = datafile.fileno()
    fileStat = os.fstat(fd)
    numRecords = int((fileStat.st_size-NLXHeaderSize)/TETRODERECORDSIZE)
    print("Number of Records: ",numRecords)
    Sweeps = np.zeros((numRecords,4,32),dtype=np.int16)
    startreading = time.time()
    for i in range(0,numRecords):
        datafile.seek(NLXHeaderSize + RecordOffset +(TETRODERECORDSIZE*i))
        insweeps = np.fromfile(datafile,dtype=np.int16,count=128)
        separateSweeps = np.reshape(insweeps,(32,4))
        Sweeps[i,0]=separateSweeps[:,0]
        Sweeps[i,1]=separateSweeps[:,1]
        Sweeps[i,2]=separateSweeps[:,2]
        Sweeps[i,3]=separateSweeps[:,3]
    endreading = time.time()

params = np.zeros((numRecords,28),dtype=np.float32)
peakTime = np.zeros(4,dtype=int)
for i in range(0,numRecords): # recreate calcparms 
    biggest = 0
    maxAmp = 0
    for s in range(0,4):
        temp = Sweeps[i,s,:]*1.0
        params[i,s]=np.sqrt(np.sum(np.square(temp)))/6      # energy (I think)
        params[i,s+4]=np.max(Sweeps[i,s,:])                 # peak
        if params[i,s+4]>biggest:                           # need this for mPeak
            maxAmp = params[i,s+4]
            biggest = s
        params[i,s+20]=np.max(Sweeps[i,s,26:31])            # late peak
        params[i,s+8]=np.min(Sweeps[i,s,:])                 # valley
        params[i,s+24]=np.min(Sweeps[i,s,26:31])            # late valley
        peakTime[s] = np.argmax(Sweeps[i,s,:])
        if peakTime[s]< 7:peakTime[s] = 7
        params[i,s+16]=np.min(Sweeps[i,s,0:peakTime[s]])    # pre valley 
        mPeaktime = np.argmax(Sweeps[i,biggest,:])          # mpeak
        if mPeaktime > 1:
            start = mPeaktime-2
        else:
            start = 0
        if mPeaktime <30:
            stop = mPeaktime +2
        else: 
            stop = 31
        params[i,s+12]= max(0,np.max(Sweeps[i,s,start:stop]))
# for i in range(0,2000):
#     whole = np.concatenate((Sweeps[i,0,:],Sweeps[i,1,:],Sweeps[i,2,:],Sweeps[i,3,:]))
#     plt.plot(whole)
# plt.show()
endreading = time.time()
print("parameters calculated: ",endreading-startreading)
numEvents = 50000
dData = np.zeros((numEvents,8),dtype=np.float32)
for i in range(0,numEvents):       # data used for analysis...
    dData[i]= params[i,0:8]         # energy and peak

# lookat = 0
# plt.scatter(params[:,lookat],params[:,lookat + 1],s=1)
# plt.show()
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))
#distances = np.zeros((numRecords,numRecords),dtype=np.float32)
startreading = time.time()
distances = cdist(dData,dData,'euclidean')
endreading = time.time()
print("distances calculated: ",endreading-startreading)
print(distances)
# for i in range(0,numRecords-1):
#     for j in range(i+1,numRecords):
#         distances[i,j]=euclidean_distance(dData[i,:],dData[j,:])
#     if i%10 ==0:
#         endreading = time.time()
#         print(i,endreading-startreading)
#         startreading = time.time()
#print ("distances calculated: ",endreading-startreading)
#print(euclidean_distance(dData[0,:],dData[1,:]))