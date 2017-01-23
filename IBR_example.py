#January Project, ILLC
#20.1.17, Silvan Hungerbuehler
#Model for scalar implication/horn game
#2X2X2, adjustable bias IBR

#GENERALIZE TO NXNXN

import numpy as np

#Boolean Matrix
b = np.array([[1,0],[1,1]], float)

sum_row_0 = b[0,0]+b[0,1]
sum_row_1 = b[1,0]+b[1,1]

trans = b.transpose()
trans_sum_row_0 = trans[0,0]+trans[0,1]
trans_sum_row_1 = trans[1,0]+trans[1,1]

#Naive Sender strategy, normalized Boolean
strat_S_0 = np.array([[0,0],[0,0]], float)
strat_S_0[0,0]=b[0,0]/sum_row_0
strat_S_0[0,1]=b[0,1]/sum_row_0
strat_S_0[1,0]=b[1,0]/sum_row_1
strat_S_0[1,1]=b[1,1]/sum_row_1



#Naive receiver strategy, normalized transposed Boolean
#WRITE FUNCTION THAT CALCULATES NAIVE STRATEGY AUTOMATICALLY
strat_R_0 = np.array([[0,0],[0,0]],float)
strat_R_0[0,0]=b.transpose()[0,0]/trans_sum_row_0
strat_R_0[0,1]=b.transpose()[0,1]/trans_sum_row_0
strat_R_0[1,0]=b.transpose()[1,0]/trans_sum_row_1
strat_R_0[1,1]=b.transpose()[1,1]/trans_sum_row_1

#Belief Level 1 Sender
bel_S_1 = np.array([[1,0],[0.5,0.5]],float)
#Belief level 1 Receiver
bel_R_1 = np.array([[0.5,0.5],[0,1]],float)


print("Strategy_S_0") #DEBUGGGING
print(strat_S_0)
print("Strategy_R_0")
print(strat_R_0)


#Best response function
def BR(array, cost = np.array([0.0,0.0]), priors = np.array([1,1])):
    '''Give best response to opponent strategy.
    :param Array containing opponents strategy
    :param Array containing cost of signals
    :param Array containing prior distribution over states
    :returns Array detailing best response
    '''

    size = array.shape #Get size of strategy matrix
    br = array.reshape(size) #Initiate corresponding best response matrix

    transp_input = array.transpose() * priors - cost #RIGHT ORDER? WE HAVE TO MAKE UP FOR PRIORS MAKING VALUES SMALLER BEFORE COST IS DEDUCTED

    if transp_input[0,0] > transp_input[0,1]:
        br[0,0] = 1
        br[0,1] = 0
    if transp_input[0,0] == transp_input[0,1]:
        br[0,0] = priors[0,0]/(priors[0,0]+priors[0,1])
        br[0,1] = priors[0,1]/(priors[0,0]+priors[0,1])
    if transp_input[0,0] < transp_input[0,1]:
        br[0,1] = 1
        br[0,0] = 0

    if transp_input[1,0] > transp_input[1,1]:
        br[1,0] = 1
        br[1,1] = 0
    if transp_input[1,0] == transp_input[1,1]: #In case there'is no strict best response we use the priors.
        br[1, 0] = priors[0, 0] / (priors[0, 0] + priors[0, 1])
        br[1, 1] = priors[0, 1] / (priors[0, 0] + priors[0, 1])
    if transp_input[1,0] < transp_input[1,1]:
        br[1,1] = 1
        br[1,0] = 0

    return br


#Testing

cost_sender = np.array([0.0,0.9])
cost_receiver = np.array([0,0])
priors1 = np.array([[0.5,0.5]],float) #TEMPTED TO FILL IN [1,1] HERE. OTHERWISE STUFF COULD GET SCREWED UP IF WE HAVE COST AND PRIORS (DOES THAT EVER HAPPEN? ARE COST STRICTLY FOR SENDERS AND PRIORS STRICTLY FOR RECEIVERS?)
priors2 = np.array([[1,0]],float)

BR_S_1 = BR(strat_R_0, cost_sender)
BR_R_1 = BR(strat_S_0, cost_receiver, priors2)

print(BR_S_1)
print(BR_R_1)


