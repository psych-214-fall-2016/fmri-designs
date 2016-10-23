# F tests in R
y1 = read.table('y1.txt')$V1
y2 = read.table('y2.txt')$V1
x = read.table('x.txt')$V1
res1 = lm(y1 ~ x)
print(summary(res1))
res2 = lm(y2 ~ x)
print(summary(res2))
