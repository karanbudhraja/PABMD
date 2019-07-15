# needed in case two results answers the same name identifier
with open("results.csv", "r") as inFile:
    with open("temp.csv", "w") as outFile:
        for line in inFile:
            line = line.replace("|", "\",\"")
            outFile.write(line)
