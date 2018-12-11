from decimal import *
import scipy
import math

def sign_test(results_1, results_2):
  """test for significance
  results_1 is a list of classification results (+ for correct, - incorrect)
  results_2 is a list of classification results (+ for correct, - incorrect)
  """
  ties, plus, minus = 0, 0, 0
  q = 0.5

  # "-" carries the error
  for i in range(0, len(results_1)):
    if results_1[i] == results_2[i]:
      ties += 1
    elif results_1[i] == 0:
      plus += 1
    elif results_2[i] == 0:
      minus += 1

  n = 2 * math.ceil(ties / 2) + plus + minus
  k = math.ceil(ties / 2) + min(plus, minus)

  summation = Decimal(0.0)
  for i in range(0, int(k) + 1):
      summation += Decimal(scipy.special.comb(n, i, exact=True))

  # use two-tailed version of test
  summation *= 2
  summation *= (Decimal(q)**Decimal(n))

  return summation
