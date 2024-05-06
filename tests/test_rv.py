import structgeo.probability as rv

rand = rv.NonRepeatingRandomListSelector(range(0,4))

for i in range(5):
    print(next(rand))

