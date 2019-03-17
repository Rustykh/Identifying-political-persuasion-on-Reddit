import sys
import argparse
import os
import json
import html
import re
import spacy
import string

indir = '/u/cs401/A1/data/';
nlp = spacy.load('en', disable=['parser', 'ner'])
#Constructing the Stopwords set
with open('/u/cs401/Wordlists/StopWords') as f:
     Stopwords = set([word.strip(' \n') for word in f.readlines()])
#Constructing the abbreviation sets
with open('/u/cs401/Wordlists/abbrev.english') as f:
    abbrevs = [ab.strip(' \n') for ab in f.readlines()]
with open('/u/cs401/Wordlists/pn_abbrev.english') as f:
    abbrevs += [ab.strip(' \n') for ab in f.readlines()]
with open('/u/cs401/Wordlists/pn_abbrev.english2') as f:
    abbrevs += [ab.strip(' \n') for ab in f.readlines()]
abbrevs = set(abbrevs)
def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    modComm = comment
    if 1 in steps:
        #Stripping newline characters and making lowercase 
        modComm = modComm.replace('\n', ' ')
        modComm = modComm.strip()
        modComm = modComm.lower()
        #print("STEP 1:" + modComm)

    if 2 in steps:
	#Removing HTML characters 
        modComm = html.unescape(modComm)
        #print("STEP 2:" + modComm)
    if 3 in steps:
	#Removing URL's 
        modComm = re.sub('(www|http)\S+', '', modComm, flags=re.MULTILINE)
        #print("STEP 3:" + modComm)
    if 4 in steps:
	#Splitting comment into tokens
        Comm = modComm.split(' ')
        no_punct = []

	#Splitting Punctuation
        for token in Comm:
	    #When there is an abbreviation
            if token in abbrevs and '.' in token:
                no_punct.append(re.sub(r'([![\\\]^_`{|}~"#$%&()*+,\/:;<=>?@\]+)', r' \1', token))
            elif '.' in token:
                no_punct.append(re.sub(r'([![\\\]^_`{|}~"#$%&()*+,.\/:;<=>?@\]+)', r' \1', token))
            else:
                no_punct.append(re.sub(r'([![\\\]^_`{|}~"#$%&()*+,\/:;<=>?@\]+)', r' \1', token))
	#rejoining the array of tokens back into a string
        modComm = ' '.join(no_punct)
        #print("STEP 4:" + modComm)


    if 5 in steps:
	#Splitting Clitics
        modComm = re.sub(r'(n\'[\w]*)', r' \1', modComm)
        modComm = re.sub(r'(?<!n)(\'[\w]*)', r' \1', modComm)
        #print("STEP 5:" + modComm)

    if 6 in steps:
	#PoS tagging using spaCy
        utt = nlp(modComm)
        tagged = []
        for token in utt:
            tagged.append(str(token.text)+'/'+str(token.tag_))
        modComm = ' '.join(tagged)
        #print("STEP 6:" + modComm)
    if 7 in steps:
        Comm = modComm.split(' ')
        CommNoStop = []
	#Removing Stop Words 
        for tagged in Comm:
            tmp = re.sub(r'\/\S+', '', tagged)
            tmp = re.sub(r'/', '', tmp)
            if tmp not in Stopwords:
                CommNoStop.append(tagged)
        modComm = ' '.join(CommNoStop)
        #print("STEP 7:" + modComm)

    if 8 in steps:
	#Applying Lemmatization
        modComm = re.sub(r'\/\S+', '', modComm)
        modComm = re.sub(r'/', '', modComm)
     
        utt = nlp(modComm)
        tagged = []
        for token  in utt:
            tagged.append(str(token.lemma_) + '/' +str(token.tag_))
        modComm = ' '.join(tagged)
        #print("STEP 8:" + modComm)


    if 9 in steps:
	#Adding newline between sentences
        modComm = re.sub(r'([!.?/]{2,})', r'\1 \n', modComm)
        #print("STEP 9:" + modComm)

    if 10 in steps:
        pass

    return modComm

def main( args ):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            filename = os.path.splitext(os.path.basename(fullFile))[0]
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            idx = args.ID[0]%(len(data))
            num_processed = 0
            while num_processed < args.max:
                line = json.loads(data[idx])
                
                line['body'] = preproc1(line['body'])
             


                line['cat'] = filename

                allOutput.append(line)
                num_processed+=1
                idx+=1
                idx=idx%len(data)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
