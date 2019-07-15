import numpy as np

# results from batch data
RESULTS_FILE = "results_flocking.csv"
INPUT_NAME ="\"Input.gifPrefix\""

def is_integer(x):
    try:
        int(x)
        return True
    except:
        return False

# main code
with open(RESULTS_FILE) as results:
    headers = results.readline()

    #print headers
    
    headers = headers.split(",")
    inputIndex = headers.index(INPUT_NAME)
    answer1Index = inputIndex + 1
    answer2Index = inputIndex + 2
    answer3Index = inputIndex + 3
    answer4Index = inputIndex + 4

    # increment all by 3 for some reason
    # to get correct values in data
    inputIndex += 3
    answer1Index += 3
    answer2Index += 3
    answer3Index += 3
    answer4Index += 3
    
    # read result data
    resultData = {}
    for line in results:
        inputData = line.split(",")[inputIndex]
        answer1 = line.split(",")[answer1Index]
        answer2 = line.split(",")[answer2Index]
        answer3 = line.split(",")[answer3Index]
        answer4 = line.split(",")[answer4Index]

        #print line
        #print inputData
        #print answer1, answer2, answer3, answer4

        if(inputData not in resultData.keys()):
            resultData[inputData] = {}
        if("answer1" not in resultData[inputData].keys()):
            resultData[inputData]["answer1"] = []
        if("answer2" not in resultData[inputData].keys()):
            resultData[inputData]["answer2"] = []
        if("answer3" not in resultData[inputData].keys()):
            resultData[inputData]["answer3"] = []

        # check control question
        if(eval(eval(answer4)) < 8):
            # control question failed
            print "control question failed: " + answer4
            continue

        # check for invalid responses
        invalidResponse = [type(eval(eval(x))) is not int for x in [answer1, answer2, answer3, answer4]]
        if(np.any(invalidResponse) == True):
            # invalid response
            print "invalid response: " + str([answer1, answer2, answer3, answer4])
        
        resultData[inputData]["answer1"].append(eval(answer1))
        resultData[inputData]["answer2"].append(eval(answer2))
        resultData[inputData]["answer3"].append(eval(answer3))

    # now process result data
    for inputData in resultData.keys():
        print inputData
        answerList = []
        for answer in resultData[inputData].keys():
            resultData[inputData][answer] = [eval(x) for x in resultData[inputData][answer] if is_integer(x)]
            #resultData[inputData][answer] = np.mean(resultData[inputData][answer])
            resultData[inputData][answer] = np.std(resultData[inputData][answer])
            print answer, resultData[inputData][answer]
            answerList.append(resultData[inputData][answer])
        print "mean: ", np.mean(answerList)
