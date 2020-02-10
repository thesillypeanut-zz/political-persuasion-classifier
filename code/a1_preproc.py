import sys
import argparse
from html import unescape
import os
import json
import re
import spacy


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

WANTED_KEYS = {'id', 'body'}


def preproc1(comment, steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
    if 2 in steps:  # unescape html
        modComm = unescape(modComm)
    if 3 in steps:  # remove URLs and returns
        modComm = re.sub(r"(http|www)\S+", "", modComm)
        modComm = re.sub(r"\r{1,}", "", modComm)
    if 4 in steps:  # remove duplicate spaces
        modComm = re.sub(" {2,}", " ", modComm)

    # remove leading and trailing spaces
    modComm = modComm.strip()

    # get Spacy document for modComm
    doc = nlp(modComm)

    # array to store the modified comment parts
    modComm = []

    for sent in doc.sents:
        for token in sent:
            # Lemmatization- replace token with its lemma if the lemma doesn't begin with "-"
            lemma = token.text if token.lemma_.startswith(
                "-") else token.lemma_
            # Tagging- Write "/POS" after each token
            modComm.append(lemma + "/" + token.tag_)
            # Split tokens with spaces
            modComm.append(" ")

        # delete the last extra space
        del modComm[-1]
        # Sentence segmentation- Insert "\n" between sentences
        modComm.append("\n")

    # return the stringified array
    return "".join(modComm)


def processComments(startIdx, endIdx, data, cat, allOutput):
    ''' This function processes a set of comments and appends them to an output array

    Parameters:     
        startIdx  : slice the data starting from this index
        endIdx    : data should be sliced up to, but not including, this index
        data      : comment dataset 
        cat       : category for the set of comments (i.e. Left, Center, Right or Alt)                                                                
        allOutput : output array that contains all processed comments
    '''

    for i in range(startIdx, endIdx):
        # load comment string
        comm = json.loads(data[i])
        # retain fields that are relevant
        comm = {key: comm[key] for key in comm if key in WANTED_KEYS}
        # process the body field and replace the 'body' field with the processed text
        comm["body"] = preproc1(comm["body"])
        # add a field to specify the comment category
        comm["cat"] = cat
        # append the result to 'allOutput'
        allOutput.append(comm)

    print(f"Processed {endIdx-startIdx} comments")


def main(args):
    allOutput = []
    id = args.ID[0]

    for subdir, _, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print(f"Processing {fullFile}...")

            data = json.load(open(fullFile))
            numComments = len(data)

            # select appropriate args.max lines
            maxComments = args.max

            # get start and end indices to process a slice of the data
            startIdx = id % numComments
            endIdx = startIdx + maxComments
            processComments(startIdx, min(endIdx, numComments),
                            data, file, allOutput)

            # handle circular indexing
            if endIdx > numComments:
                endIdx = endIdx - numComments if maxComments <= numComments else startIdx
                if endIdx > 0:
                    # process comments from the beginning of the dataset
                    processComments(0, endIdx, data, file, allOutput)

            print(f"✓ Completed\n")

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument(
        "-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument(
        "--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument(
        "--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')

    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    indir = os.path.join(args.a1_dir, 'data')
    main(args)
