import numpy as np
from scipy.stats import triang, lognorm, pareto

#Task1
#from dhCheck_Task1 import dhCheckCorrectness
def Task1(a, b, c, point1, point2, data, mu, sigma, xm, alpha,
num, point3, point4, point5):
    # ALE = ARO X SLE
    # SLE = AV X EF

    # create a triangular distribution
    tridis = triang(c = (c-a)/(b-a), loc = a, scale = b-a)

    # prob 1
    # F(x) = [(x-𝒂)²]/[(𝒃-𝒂)(𝒄-𝒂)] for 𝒂 ≤ x < 𝒄
    prob1 = tridis.cdf(point1)
    print("Prob1 is ",prob1)

    # prob 2
    # F(x) = 1 - [(𝒃-x)²]/[(𝒃-𝒂)(𝒃-𝒄)] for 𝒄 ≤ x ≤ 𝒃
    prob2 = tridis.cdf(point2)
    print("Prob2 is ", prob2)

    # mean_t
    MEAN_t = tridis.mean()
    print("MEAN_t is ", MEAN_t)

    # median_t
    MEDIAN_t = tridis.median()
    print("MEDIAN_t is ", MEDIAN_t)

    # dataset
    # mean_d
    MEAN_d = np.mean(data)
    print("MEAN_d is", MEAN_d)

    # variance_d
    VARIANCE_d = np.var(data, ddof=0)
    print("VARIANCE_d is ", VARIANCE_d)

    # flaw A
    # Log Norm distribution
    lognorm = np.random.lognormal(mean=mu, sigma= sigma, size=num)

    # flaw B
    # Pareto distribution
    pareto = np.random.pareto(alpha, size=num) + xm

    # total impact - sum of impacts caused by both flows
    totalImpact = lognorm + pareto
    print("total impact is ", totalImpact)

    # prob 3
    prob3 = np.sum(totalImpact > point3) / num
    print("Prob3 is ", prob3)

    # prob 4
    prob4 = np.sum((totalImpact >= point4) & (totalImpact <= point5)) / num
    print("Prob4 is ", prob4)

    # ALE
    # AV = MEDIAN_t
    # ARO = MEAN_d
    # EF = prob3
    SLEex = MEDIAN_t * prob3
    ALE = MEAN_d * SLEex
    print("ALE is ", ALE)


    return (prob1, prob2, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d,
prob3, prob4, ALE)


# Task2
# from dhCheck_Task2 import dhCheckCorrectness
def Task2(num, table, eventA, eventB, probs):
    # Prob1 with bayes, eventA with x = 7
    data = np.array(table)
    column = data[:,eventA]
    columnsum = np.sum(column)
    prob1 = columnsum / num
    print("Prob1 is ", prob1)

    # Prob2 with bayes, eventB
    row1 = data[:][eventB[0]]
    row1sum = np.sum(row1)
    row2 = data[:][eventB[1]]
    row2sum = np.sum(row2)
    rows = np.sum(row1sum + row2sum)
    prob2 = rows / num
    print("Prob2 is ", prob2)
    
    # IsInd
    intersection = prob1 * prob2
    #print("Intersection is", intersection)
    union = prob1 + prob2 - (prob1*prob2)
    #print("Union is", union)

    bayes = intersection / prob2
    print(bayes)
    if intersection == union:
        IsInd = 1
    else:
        IsInd = 0
    print(IsInd)

    # Prob3
    pt = np.sum(probs)
    prob3 = pt / len(probs)
    #prob3 = np.sum(probs) / (num - np.sum(data[:,3]))
    #print(num - np.sum(data[:,3]))
    print("Prob3 is ", prob3)

    # Prob4
    
    #prob4 = np.sum(data[:,3]) / (num - np.sum(data[:,3]))
    #prob4 = 1 / np.sum(data[:,3])
    print("Prob4 is", prob4)

    return (prob1, prob2, IsInd, prob3, prob4)

if __name__ == "__main__":

    # TASK 1 
    """
    a = 10000
    b = 35000
    c = 18000
    point1, point2 = 12000, 25000
    mu, sigma, xm, alpha, num = 0,3,1,4,500000
    point3, point4, point5 = 30, 50, 100
    data = [11, 15, 9, 5, 3, 14, 16, 15, 12, 10, 11, 4, 7, 12, 6]
    prob1, prob2, MEAN_t, MEDIAN_t, MEAN_d, VARIANCE_d,prob3, prob4, ALE = Task1(a, b, c, point1, point2, data, mu, sigma, xm, alpha,
num, point3, point4, point5)
    
    """
    # TASK 2
    num = 120
    eventA = 2 #column 2 of table
    eventB = [0,1] #row 0 and row 1 of table
    probs = [0.7, 0.6, 0.5, 0.63, 0.44, 0.36]
    table = [[6, 10, 11, 9], [9, 12, 15, 8], [7, 14, 10, 9]]

    prob1, prob2, IsInd, prob3, prob4 = Task2(num, table, eventA, eventB, probs)
    