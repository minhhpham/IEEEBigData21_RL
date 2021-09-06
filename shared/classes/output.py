# function to save results to output
# format data according to submission format and write to file
def writeOutput(finalItemSet, outFileName, userIDs_, outDir = '/tf/shared/outputs/'):
    assert len(finalItemSet)==len(userIDs_)
    outFile = outDir + outFileName
    f = open(outFile, "w")
    f.write('id,itemids')
    for i in range(len(userIDs_)):
        f.write('\n')
        itemList = finalItemSet[i]
        itemString = ' '.join([str(j) for j in itemList])
        outString = str(userIDs_[i]) + ',' + itemString
        f.write(outString)