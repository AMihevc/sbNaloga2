abc = [(1,1,1),(1,0,1),(0,0,0),(1,1,1)]
res = (0,0,0)

for a in abc:
    res = tuple(map(sum, zip(a, res)))  

gdf = (1,2,3)

print(sum(gdf))