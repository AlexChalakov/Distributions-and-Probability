import numpy as np
from scipy.stats import triang, lognorm, pareto
from scipy.optimize import curve_fit

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
    #print(bayes)
    if intersection == union:
        IsInd = 1
    else:
        IsInd = 0
    print(IsInd)

    # Prob3
    # Note down the positives
    condY3T = 0.7
    condY4T = 0.6
    condY5T = 0.5
    condX5T = 0.63
    condX6T = 0.44
    condX7T = 0.36

    # Get the probability of rows
    rowY3 = data[:][0]
    rowY3sum = np.sum(rowY3)
    probY3 = rowY3sum / num

    rowY4 = data[:][1]
    rowY4sum = np.sum(rowY4)
    probY4 = rowY4sum / num

    rowY5 = data[:][2]
    rowY5sum = np.sum(rowY5)
    probY5 = rowY5sum / num

    # Get the intersections
    # P=(Y=... n T)
    intY3 = condY3T * probY3
    intY4 = condY4T * probY4
    intY5 = condY5T *probY5

    # Calculate total probability
    prob3  = intY3 + intY4 + intY5
    print("Prob3 is ", prob3)

    # Prob4
    # Do the same thing for the columns like it was done for the rows
    columnX5 = data[:,0]
    columnX5sum = np.sum(columnX5)
    probX5 = columnX5sum / num

    columnX6 = data[:,1]
    columnX6sum = np.sum(columnX6)
    probX6 = columnX6sum / num

    columnX7 = data[:,2]
    columnX7sum = np.sum(columnX7)
    probX7 = columnX7sum / num

    columnX8 = data[:,3]
    columnX8sum = np.sum(columnX8)
    probX8 = columnX8sum / num

    # Intersections agains
    intX5 = condX5T * probX5
    intX6 = condX6T * probX6
    intX7 = condX7T * probX7

    # Bayes Theorem
    intX8 = prob3 - (intX5 + intX6 + intX7)
    reverseX8 = intX8 / probX8
    prob4 = (reverseX8 * probX8)/ prob3
    print("Prob4 is", prob4)

    return (prob1, prob2, IsInd, prob3, prob4)

#from dhCheck_Task3 import dhCheckCorrectness
def Task3(x, y, z, num1, num2, num3, num4, bound_y, bound_z, c,
se_bound, ml_bound, x_bound, x_initial):
    # TODO
    # Linear regression using curve_fit from workshop
    # weights_b, weights_d
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    def fn(xdata, b0, b1, b2, b3, b4, b5):
        return b0 + b1*xdata[0] + b2*xdata[1] + b3*xdata[2] + b4*xdata[3] + b5*xdata[4]
    weights_b, pcov = curve_fit(fn, x, y)
    weights_d, pcov = curve_fit(fn, x, z)
    print(weights_b, weights_d)

    #10 historical pairs
    # s_num5, l_num5
    pairs = 10
    safeguard = weights_b[0] + num1*weights_b[1] + num2*weights_b[2] + num3*weights_b[3] + num4*weights_b[4]
    maintenance = weights_d[0] + num1*weights_d[1] + num2*weights_d[2] + num3*weights_d[3] + num4*weights_d[4]

    #print(safeguard)
    for s_num5 in range(pairs + 1):
        if safeguard + s_num5*weights_b[5] >= bound_y:
                s_num5 = safeguard + s_num5*weights_b[5]
                print(s_num5)

    # x_add
    # use linprog - create the parameters beforehand
    Aie = [safeguard, maintenance]

    


    return (weights_b, weights_d, s_num5, l_num5, x_add)

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
    """
    num = 120
    eventA = 2 #column 2 of table
    eventB = [0,1] #row 0 and row 1 of table
    probs = [0.7, 0.6, 0.5, 0.63, 0.44, 0.36]
    table = [[6, 10, 11, 9], [9, 12, 15, 8], [7, 14, 10, 9]]

    prob1, prob2, IsInd, prob3, prob4 = Task2(num, table, eventA, eventB, probs)
    """
    # TASK 3
    x = [[5,4,8,8,2,5,5,7,8,8],[3,7,7,2,2,5,10,4,6,3],[8,3,6,7,9,10,6,2,2,3],[9,3,9,3,10,4,2,3,7,5],
    [4,9,6,6,10,3,8,8,4,6]]
    y = [176,170,215,146,228,145,183,151,160,151]
    z = [352,384,471,358,412,345,449,357,366,349]
    num1, num2, num3, num4, bound_y, bound_z = 5, 6, 8, 4, 160, 600
    c = [11, 6, 8, 10, 9]
    se_bound = 1000
    ml_bound = 2000
    x_bound = [30,50,20,45,50]
    x_initial = [3,5,4,2,1]

    weights_b, weights_d, s_num5, l_num5, x_add = Task3(x, y, z, num1, num2, num3, num4, bound_y, bound_z, c,
se_bound, ml_bound, x_bound, x_initial)
    