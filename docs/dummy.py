# dummy program that checks for substrings

PRIME = 2 ** 31 - 1

def search_lyrics(L, Q):
    """
    Input: L | an ASCII string 
    Input: Q | an ASCII string where |Q| < |L|
    
    Return `True` if Q appears inside the lyrics L and `False` otherwise.
    """

    ##################
    # YOUR CODE HERE #

    string_size = len(Q)
    all_strings = []

    for i in range(len(L)-len(Q)):
        all_strings.append(L[i:i+len(Q)])

    list_of_Rs = {}

    ## start with calculating R(L1)
    R = 0
    scale_multiplier = 1
    for i in range(string_size - 1, -1, -1):
        ascii_val = ord(all_strings[0][i])
        R += ascii_val*scale_multiplier % PRIME
        if i != 0:
            scale_multiplier = scale_multiplier * 128
    R = R % PRIME
    f_dash = scale_multiplier % PRIME 
    list_of_Rs[R] = True
    R_prev = R


    for i in range(1, len(all_strings)):
        most_significant_value = f_dash * ord(all_strings[i-1][0])
        R_new = ((R_prev - most_significant_value) * 128) % PRIME
        R_new = R_new + ord(all_strings[i][-1])
        R_prev = R_new
        R_new = R_new % PRIME
        list_of_Rs[R_new] = True
    
    R_Q = 0
    scale_multiplier = 1
    for i in range(string_size - 1, -1, -1):
        ascii_val = ord(Q[i]) 
        R_Q += ascii_val*scale_multiplier
        scale_multiplier = scale_multiplier * 128
    R_Q = R_Q % PRIME
    
    if R_Q in list_of_Rs:
        return True
    else:
        return False




    ##################

    #return False

