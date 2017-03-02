from sys import argv
from scipy.stats import norm

n = int(argv[1])
#s = int(argv[2])
#assert s <= n
d = 0.98
e = 0.2

weighted_score = 0
for i in xrange(n):
    weighted_score -= d ** (2 * n - i)
    #weighted_score += (e * 2 * (d ** (2 * n - 1))) + ((1 - e) * (-1) * (d ** (2 * n - 1)))
print weighted_score
s = (n + abs(weighted_score)) / 3.0
print s
var = e * (1 - e) / n
std = var ** 0.5
x = s / n
prob = 1 - norm.cdf((x - e) / std)
print prob
print 1 / prob

s2 = (2 * n + abs(weighted_score)) / 4.0
x2 = s2 / n
prob2 = 1 - norm.cdf((x2 - e) / std)
print prob2
print 1 / prob2
'''
for i in xrange(s):
    weighted_score += 2 * (d ** (n - i))
#print weighted_score
for i in xrange(n - s):
    weighted_score -= d ** (n - s - i)
#print weighted_score
'''
