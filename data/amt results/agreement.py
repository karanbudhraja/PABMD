import nltk
import sys
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import interval_distance

inputFileName = sys.argv[1]
BLOCK_SIZE = eval(sys.argv[2])

with open(inputFileName, 'r') as inFile:
    lines = inFile.readlines()

    header = lines[0].split(',')
    answerIndexes = [index for index in range(len(header)) if 'answer' in header[index]]

    allAnswers = []
    for line in lines[1:]:
        answers = [line.split(',')[answerIndex+3] for answerIndex in answerIndexes]
        try:
            answers = [eval(x) for x in answers]
            allAnswers.append(answers)            
        except:
            # invalid line
            pass

    print 'total answers:', len(allAnswers)
    numberOfBlocks = int(len(allAnswers)/BLOCK_SIZE)
    print 'number of blocks:', numberOfBlocks

    for block in range(numberOfBlocks):
        currentIndexStart = block*BLOCK_SIZE
        currentIndexStop = (block+1)*BLOCK_SIZE
        currentBlockAnswers = allAnswers[currentIndexStart:currentIndexStop]
        
        allTriples = []
        for annotatorIndex in range(len(currentBlockAnswers)):
            answers = currentBlockAnswers[annotatorIndex]
            for taskIndex in range(len(answers)):
                triple = (str(annotatorIndex), str(taskIndex), eval(str(answers[taskIndex])))

                if type(triple[-1]) == dict:
                    # no data
                    pass
                else:
                    allTriples.append(triple)
                
        t = AnnotationTask(allTriples, interval_distance)
        print 'alpha', t.alpha()
