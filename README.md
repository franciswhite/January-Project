# January-Project

# Running the IBR-model
B is the given semantical definition of the messages
 
    B = np.array([[1, 1, 0, 0],
                 [0, 0, 1, 1]])
                 
Sender cost is the given by B.shape[1]

    sender_cost = np.array([0, 1, 0, 0])
    
Receiver cost is the given by B.shape[1]

    receiver_cost = np.array([0, 0, 0, 1])
    
Priors are given by B.shape[0]

    priors = np.array([0.5, 0.5])

Then construct the object IteratedModel(B, sender_cost, receiver_cost, priors)

    iterated_model = IteratedModel(B, sender_cost, receiver_cost, priors)
    
And run until solved (an automatic detection of a stable strategy is not implemented)

    for x in range(0, 5):
        iterated_model.next_level_reasoning()
    # to see solution
    print(iterated_model)