with open("Report/bib.txt") as fp:
    Lines = fp.readlines()
    #print(len(Lines))
    count = 0
    for line in Lines:
        count +=1
        print("{}: \cite{{{}}},".format(count, line.strip()))