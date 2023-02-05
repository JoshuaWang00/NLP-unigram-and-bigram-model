
#open training and test data while making it lowercase and adding start+end tags
import math

with open('train-Fall2022.txt', 'r', encoding ='utf-8') as trainText:
    corpusList = [x.rstrip() for x in trainText]

corpusList = ['<s> ' + item.lower() + ' </s>' for item in corpusList]
#open test data similarly to training data
with open('test.txt', 'r', encoding ='utf-8') as testext:
    testList = [x.rstrip() for x in testext]

testList = ['<s> ' + item.lower() + ' </s>' for item in testList]

trainingVocab = []
testVocab = []
#split training lines into individual word tokens seperated by whitespace
for x in corpusList:
	tempList = x.split()
	trainingVocab.extend(tempList)
#split test data
for x in testList:
	tempList = x.split()
	testVocab.extend(tempList)
#track total number of word types using set to remove dupes
trainingVocabSet = set(trainingVocab)
testVocabSet = set(testVocab)
#use dictionary to assign words to their count for both corpuses
trainDict = {}
testDict = {}
for x in trainingVocab:
	if x in trainDict:
		trainDict[x]+=1
	else:
		trainDict.update({x:1})

for x in testVocab:
	if x in testDict:
		testDict[x]+=1
	else:
		testDict.update({x:1})
#check dictionary for counts of 1 to assign unknown token to training
finalTrainingVocab = []
for x in trainingVocab:
	if trainDict[x] > 1:
		finalTrainingVocab.append(x)
	else:
		finalTrainingVocab.append("<unk>")

#convert to set to remove dupes, turns tokens to types for q1
finalTrainingVocabSet = (set(finalTrainingVocab))
#replace test data not seen in training with unknown tag
finalTestVocab = []
for x in testVocab:
	if x in finalTrainingVocabSet:
		finalTestVocab.append(x)
	else:
		finalTestVocab.append("<unk>")


#count of word tokens in training dictionary with unknown tags included
finalTrainDict = {}
for x in finalTrainingVocab:
	if x in finalTrainDict:
		finalTrainDict[x]+=1
	else:
		finalTrainDict.update({x:1})

#convert test and training data to bigrams, skip over <s> so it is not considered first in sentence
trainPre = ''
testPre = ''
trainingBigram = []
testBigram = []
for x in finalTrainingVocab:
	if x == "<s>":
		trainPre = "<s>"
	else:
		trainingBigram.append(trainPre + " " + x)
		trainPre = x

for x in finalTestVocab:
	if x == "<s>":
		testPre = "<s>"
	else:
		testBigram.append(testPre + " " + x)
		testPre = x
#since we will need bigram counts for bigram probability calculations, set up dictionaries to track bigram counts
testBigramDict = {}
trainBigramDict = {}
for x in trainingBigram:
	if x in trainBigramDict:
		trainBigramDict[x]+=1
	else:
		trainBigramDict.update({x:1})

for x in testBigram:
	if x in testBigramDict:
		testBigramDict[x]+=1
	else:
		testBigramDict.update({x:1})


#for q1 to find how many unique words there are without <s>
print("Q1:Word Tokens: " + str(len(finalTrainingVocabSet)-1))
#for q2 to find how many word tokens there are excluding <s>
print(("Q2:Word Types: ") + str(len(finalTrainingVocab)-trainDict["<s>"]))
#for q3 to find percentage of test word tokens and word types not in training data before <unk> tagging and excluding <s>
#can be done if we subtract from 1 the matching percentage of test data
#find intersecting word types using set
matchingTypes = trainingVocabSet.intersection(testVocabSet)
matchingTypes.remove("<s>")
matchTypeCount = len(matchingTypes)
#to find intersecting word tokens, we use our matchingTypes set, count that total amount in our test data
matchTokenCount = 0
for x in testVocab:
	if x in matchingTypes:
		matchTokenCount+=1
print("Q3:Non Intersecting Test Types: " + str(1-(matchTypeCount/(len(testVocabSet)-1))))
print("Q3:Non Intersecting Test Tokens: " + str(1-(matchTokenCount/(len(testVocab)-testDict["<s>"]))))
#for q4 to find percentage of bigrams in test corpus but not in training
#found similarly to q3, first find percentage of matching out of test vocab, subtract that from 1
matchingBigrams = set(trainingBigram).intersection(set(testBigram))
matchBigramCount = len(matchingBigrams)
#for q4 types
print("Q4:Non Intersecting Bigram Test Types: " + str(1-(matchBigramCount/len(testBigram))))
#for q4 tokens
matchBigramTokenCount = 0
for x in testBigram:
	if x in matchingBigrams:
		matchBigramTokenCount+=1
print("Q4:Non Intersecting Bigram Test Tokens: " + str(1-(matchBigramTokenCount/len(testBigram))))
#for q5 log probabilities
#unigram model
totalTrainingTokens = len(finalTrainingVocab)-trainDict["<s>"]
#we also want to check if our sentence has any words not seen in training that need to be unknown
sentence = "<s> i look forward to hearing your reply . </s>"
splitSentence = sentence.split()
mappedSentence = []
for x in splitSentence:
	if x in finalTrainingVocabSet:
		mappedSentence.append(x)
	else:
		mappedSentence.append("<unk>")
#result of our sentence processing reveals every word in our sentence has been seen in training

unigramSentenceProb = 0
for x in mappedSentence:
	if x != "<s>":
		unigramSentenceProb += math.log2(finalTrainDict[x]/totalTrainingTokens)
print("Q5:Unigram Sentence Probability: " + str(unigramSentenceProb))

#bigram model
bigramSentence = []
bigPre = ''
for x in mappedSentence:
	if x == "<s>":
		bigPre = "<s>"
	else:
		bigramSentence.append(bigPre + " " + x)
		bigPre = x

#our bigram calculation unfortunately gets no result since the bigram "hearing your" cant be found in our training dictionary
#bigramSentenceProb = 0
#for x in bigramSentence:
#	bigramSentenceProb += math.log2(trainBigramDict[x]/trainDict[x.split()[0]])
#print(bigramSentenceProb)

#bigram add one smoothing model
bigramSentenceProb = 0
vocabSize = len(finalTrainingVocabSet) - 1
for x in bigramSentence:
	if x in trainBigramDict:
		bigramSentenceProb += math.log2((trainBigramDict[x]+1)/(finalTrainDict[x.split()[0]]+vocabSize))
	else:
		bigramSentenceProb += math.log2(1/(finalTrainDict[x.split()[0]]+vocabSize))
print("Q5:Bigram Smoothed Sentence Probability: " + str(bigramSentenceProb))

#Q6 sentence perplexity, 9 is the amount of sentence tokens except <s>
print("Q6:Unigram Sentence Per Token Prob.: " + str((unigramSentenceProb/9)))
print("Q6:Unigram Sentence Perplexity: " + str(2**-(unigramSentenceProb/9)))
print("Q6:Bigram Sentence Per Token Prob.: " + str((bigramSentenceProb/9)))
print("Q6:Bigram Sentence Perplexity: " + str(2**-(bigramSentenceProb/9)))
#q7 test corpus perplexity
#unigram log probability calculation
unigramCorpusProb = 0
for x in finalTestVocab:
	if x != "<s>":
		unigramCorpusProb += math.log2(finalTrainDict[x]/totalTrainingTokens)
#get the test token size excluding <s>
corpusSize = len(finalTestVocab) - testDict["<s>"]
print("Q7:Unigram Test Corpus Probability: " + str(unigramCorpusProb))
print("Q7:Training Corpus Length: " + str(totalTrainingTokens))
print("Q7:Test Corpus Length: " + str(corpusSize))
print("Q7:Unigram Test Corpus Per Token Prob.: " + str(unigramCorpusProb/corpusSize))
print("Q7:Unigram Test Corpus Perplexity: " + str(2**-(unigramCorpusProb/corpusSize)))
#bigram smoothed log probability calculation
bigramCorpusProb = 0
for x in testBigram:
	if x in trainBigramDict:
		bigramCorpusProb += math.log2((trainBigramDict[x]+1)/(finalTrainDict[x.split()[0]]+vocabSize))
	else:
		bigramCorpusProb += math.log2(1/(finalTrainDict[x.split()[0]]+vocabSize))
print("Q7:Bigram Test Corpus Probability: " + str(bigramCorpusProb))
print("Q7:Bigram Test Corpus Per Token Prob.: " + str(bigramCorpusProb/corpusSize))
print("Q7:Bigram Test Corpus Perplexity: " + str(2**-(bigramCorpusProb/corpusSize)))
