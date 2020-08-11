# Cross Entropy is widely used as a loss function in classification problem (e.g. CNN)
from matplotlib import pyplot
from math import log2

# determine two discrete random variables P & Q
K = range(3)
P = [1/3,1/3,1/3]
Q = [1/6,1/3,1/3]

pyplot.subplot(2,1,1)
pyplot.bar(K,P)
pyplot.subplot(2,1,2)
pyplot.bar(K,Q)
pyplot.show()

# calculate Entropy
def calculateEntropy( prob ):
    return -sum([prob[i]*log2(prob[i]) for i in range(len(prob))])

print("--- Entropy ---")

HP = calculateEntropy(P)
print("H(P): %.3f bits" % HP)

HQ = calculateEntropy(Q)
print("H(Q): %.3f bits" % HQ)

# calculate Cross Entropy
# Cross Entropy is a metric to describe the distance between the probability of P and Q
def calculateCrossEntropy( prob1, prob2 ):
    return -sum([prob1[i]*log2(prob2[i]) for i in range(len(prob1))])

print("--- Cross Entropy ---")

# Total average information of Q from the probability P
# (How far is Q in the view of P)
HPQ = calculateCrossEntropy( P, Q )
print("H(P,Q): %.3f bits" % HPQ)

# Total average information of P from the probability Q
# (How far is P in the view of Q)
HQP = calculateCrossEntropy( Q, P )
print("H(Q,P): %.3f bits" % HQP)

# Note that Cross Entropy is not symmetric, i.e. HPQ ~= HQP. This is because that
# the base is different (the point of view)
