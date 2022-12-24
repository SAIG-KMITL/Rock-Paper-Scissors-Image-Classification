with open('config.txt') as f:
    contents = f.readlines()
    con = contents[0].split(" ")
    numscale = int(con[4].replace("\n", ""))
    print(numscale)
