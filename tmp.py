t = 'kfdjakfjaldfjlakjdfl'
t = list(t)
print(t)


n = 4
examples = list(range(21))
batches = [examples[i: i+n] for i in range(0, len(examples), n)]

for b in batches:
    print(b)