class Utils:
    
    def __init__(self):
        pass
    
    @staticmethod
    def ccdf(values):
        from itertools import groupby
        x = []
        y = []
        values = sorted(values)

        # First make dist
        dist = [(key, len(list(group))) for key, group in groupby(values)]

        # Then compute inverse cumulative
        total = 1.0
        for (val, count) in dist:
            x.append(val)
            y.append(total)
            total -= count/len(values)
        return x, y
