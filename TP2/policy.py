import numpy as np

def UCB1(T, MAB, rho=0.2, print_info=False, naive=False):
    # Init:
    nb_arms = len(MAB)
    rew = np.zeros(T)
    draws = np.zeros(T).astype(int)
    N = np.zeros(nb_arms).astype(int)
    S = np.zeros(nb_arms)
    for t in range(min(T, nb_arms)):
        rew[t] = MAB[t].sample()
        draws[t] = t
        N[t] += 1
        S[t] += rew[t]
    
    for t in range(nb_arms,T):
        # for each arm, compute the score B
        if naive:
            B = S / N
        else:
            B = S / N + rho * np.sqrt(np.log(t)/(2*N))
        # select next arm
        a_next = np.argmax(B)
        # compute the reward
        r = MAB[a_next].sample()[0]
    
        # Update S and N:
        draws[t] = a_next
        rew[t] = r
        N[a_next] += 1
        S[a_next] += r
        
        if print_info:
            print("t = {}".format(t))
            print("Next Arm to draw: {}".format(a_next + 1))
            print("Reward of the next arm drawn: {}".format(r))
            print("N updated: {}".format(N))
            print("S updated: {}".format(S))
            print("\n")

    return rew,draws

def TS(T, MAB, print_info=False, adaptation=False):
    # init:
    nb_arms = len(MAB)
    rew = np.zeros(T)
    draws = np.zeros(T).astype(int)
    N = np.zeros(nb_arms).astype(int)
    S = np.zeros(nb_arms)
    for t in range(T):
        # compute beta distribution and sample a mu from this distribution
        mu = np.random.beta(S+1, N-S+1)
        a_next = np.argmax(mu)
        r = MAB[a_next].sample()[0]
        
        # For part 1.2 and question 2
        if adaptation:
            r = np.random.binomial(1,r)
            
        draws[t] = a_next
        rew[t] = r
        N[a_next] += 1
        S[a_next] += r
        
        if print_info:
            print("t = {}".format(t))
            print("Next Arm to draw: {}".format(a_next + 1))
            print("Reward of the next arm drawn: {}".format(r))
            print("N updated: {}".format(N))
            print("S updated: {}".format(S))
            print("\n")
    return rew,draws