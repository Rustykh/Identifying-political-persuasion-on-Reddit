import numpy as np
import sys
import argparse
import os
import json
import re
import string
import csv


#Initialising global variables
with open('/u/cs401/Wordlists/First-person') as f:
    first_person = set([word.strip('\n') for word in f.readlines()])
with open('/u/cs401/Wordlists/Second-person') as f:
    second_person = set([word.strip('\n') for word in f.readlines()])
with open('/u/cs401/Wordlists/Third-person') as f:
    third_person = set([word.strip('\n') for word in f.readlines()])
BGL = {}
War = {}
with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv') as f:
    rows = csv.reader(f)
    for row in rows:
        if row[0]=='Source' or row[0]=='':
            continue
        BGL[row[1].lower()] = (float(row[3]), float(row[4]), float(row[5]))
with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv') as f:
    rows = csv.reader(f)
    for row in rows:
        if row[2]=='V.Mean.Sum' or row[0]=='':
            continue
        War[row[1].lower()] = (float(row[2]), float(row[5]), float(row[8]))
slang = set(['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff', 'wyd',
'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl',
'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys', 'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr',
'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'])


center_feats = np.load('/u/cs401/A1/feats/Center_feats.dat.npy')
alt_feats = np.load('/u/cs401/A1/feats/Alt_feats.dat.npy')
right_feats =  np.load('/u/cs401/A1/feats/Right_feats.dat.npy')
left_feats =  np.load('/u/cs401/A1/feats/Left_feats.dat.npy')
center_ID = open('/u/cs401/A1/feats/Center_IDs.txt', 'r').read().split('\n')
alt_ID = open('/u/cs401/A1/feats/Alt_IDs.txt', 'r').read().split('\n')
right_ID = open('/u/cs401/A1/feats/Right_IDs.txt', 'r').read().split('\n')
left_ID = open('/u/cs401/A1/feats/Left_IDs.txt', 'r').read().split('\n')
cats = {'Left':(0,left_feats, left_ID), 'Center':(1,center_feats, center_ID), 'Right':(2, right_feats, right_ID), 'Alt':(3, alt_feats, alt_ID)}

#Helper function that counts the three types of pronouns given an array of sentences.
def count_pronouns(sentences):
    first, second, third = 0, 0, 0
    for sentence in sentences:
        sentence = sentence.split(' ')
        for token in sentence:
            word = re.sub(r'\/\S+', '', token)
            if word in first_person:
                first+=1
            elif word in second_person:
                second+=1
            elif word in third_person:
                third+=1
    return first, second, third

#Helper function that counts the slang words in an array of sentences
def count_slang(sentences):
    count = 0
    for sentence in sentences:
        sentence = sentence.split(' ')
        for token in sentence:
            word = re.sub(r'\/\S+', '', token)
            if word in slang:
                count+=1
    return count

#Helper functions to count average length of sentences in tokens,
#average token length, and number of sentences
def count_tokens_sentences(sentences):
    count = 0.0
    num_tokens = 0.0
    avg_length = 0.0
    num_tokens_not_punct = 0.0
    for sentence in sentences:
        if sentence!='':
            count+=1
            sentence = sentence.split(' ')

            for token in sentence:
                word = re.sub(r'\/\S+', '', token)
                if word!='':
                	#avg_length+=len(word)
                	num_tokens+=1
                	if word[0] not in string.punctuation:
                		avg_length+=len(word)
                		num_tokens_not_punct+=1
                		#print(word, avg_length, num_tokens_not_punct)
    if count==0:
     	   if num_tokens_not_punct==0:
     	   	return 0, 0, count
     	   else:
     	   	return 0, float(avg_length/num_tokens_not_punct), count
    if num_tokens_not_punct==0:
     	   return float(num_tokens/count), 0, count




    return float(num_tokens/count), float(avg_length/num_tokens_not_punct), count

#Helper function to get the Bristol, Gilhooly and Logie norms
def get_AoA_IMG_FAM(sentences):
    AoA, IMG, FAM = [], [], []
    for sentence in sentences:
        sentence = sentence.split(' ')
        for token in sentence:
            word = re.sub(r'\/\S+', '', token)
            if word in BGL:
                AoA.append(BGL[word][0])
                IMG.append(BGL[word][1])
                FAM.append(BGL[word][2])
    return AoA, IMG, FAM

#Helper function to get the Warringer norms for an array of sentences.
def get_V_A_D(sentences):
    V, A, D = [], [], []
    for sentence in sentences:
        sentence = sentence.split(' ')
        for token in sentence:
            word = re.sub(r'\/\S+', '', token)
            if word in War:
                V.append(War[word][0])
                A.append(War[word][1])
                D.append(War[word][2])
    return V, A, D

def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feat = np.zeros(29)
    sentences = comment.split('\n')

    #Count pronouns
    feat[0], feat[1], feat[2] = count_pronouns(sentences)

    #Count all instances of future tense verbs
    feat[5] = len(re.findall(r'((\'ll)|(will)|(going\/\w+ to\/\w+ \w+\/VB)|(gonna))', comment))

    #Count coordinating conjunctions, past tense verbs, and commas
    feat[3], feat[4], feat[6]= len(re.findall(r'\/CC ', comment)), len(re.findall(r'\/VBD ', comment)), len(re.findall(r'\/,', comment))

    #Count instances of multiple punctuation
    feat[7] = len(re.findall(r'[!?.]{2,}',  comment))

    #Count common nouns, proper nouns, adverbs and wh- words.
    feat[8] = len(re.findall(r'(\/NN |\/NNS)', comment))
    feat[9] = len(re.findall(r'(\/NNP|\/NNPS)', comment))
    feat[10] = len(re.findall(r'(\/RB|\/RBR|\/RBS)', comment))
    feat[11] = len(re.findall(r'(\/WDT|\/WP|\/WP$|\/WRB)', comment))

    #Count slang
    feat[12] = count_slang(sentences)

    #Count average length of sentences in tokens, average token length, and number of sentences
    feat[14], feat[15], feat[16]= count_tokens_sentences(sentences)

    #Getting the Bristol, Gilhooly and Logie norm values
    AoA, IMG, FAM = get_AoA_IMG_FAM(sentences)

    #Getting the Warringer norm values
    V, A, D = get_V_A_D(sentences)

    #Computing the mean and standard deviation of all the norm values
    if len(AoA)>0:
        feat[17] = np.mean(AoA)
        feat[20] = np.std(AoA)
    if len(IMG)>0:
        feat[18] = np.mean(IMG)
        feat[21] = np.std(IMG)
    if len(FAM)>0:
        feat[19] = np.mean(FAM)
        feat[22] = np.std(FAM)
    if len(V)>0:
        feat[23] = np.mean(V)
        feat[26] = np.std(V)
    if len(A)>0:
        feat[24] = np.mean(A)
        feat[27] = np.std(A)
    if len(D)>0:
        feat[25] = np.mean(D)
        feat[28] = np.std(D)
    return feat










def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    for i in range(len(data)):
    	print(i)
    	feats[i][:29] = extract1(data[i]['body'])

        #Getting the LIWC/receptiviti features
    	idx = cats[data[i]['cat']][2].index(data[i]['id'])
    	feats[i][29:-1] = cats[data[i]['cat']][1][idx][:]

        #Setting the category value(The target value for training)
    	feats[i][-1] = cats[data[i]['cat']][0]


    np.savez_compressed( args.output, feats)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()


    main(args)
