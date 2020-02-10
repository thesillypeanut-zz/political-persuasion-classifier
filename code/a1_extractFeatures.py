import csv
import numpy as np
import argparse
import json
import os
import re

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
PUNCTUATION_TAGS = {"``", "''", ",", "-LRB-",
                    "-RRB-", ".", ":", "HYPH", "NFP", "$", "SYM"}

# first 29 features and their indices
UPPERCASE = 0
FIRST_PERSON_PRONOUN = 1
SECOND_PERSON_PRONOUN = 2
THIRD_PERSON_PRONOUN = 3
COORDINATING_CONJUCTION = 4
PAST_TENSE_VERB = 5
FUTURE_TENSE_VERB = 6
COMMA = 7
MULTI_CHAR_PUNCTUATION = 8
COMMON_NOUN = 9
PROPER_NOUN = 10
ADVERB = 11
WH_WORD = 12
SLANG_ACRONYM = 13
AVG_LEN_SENTENCES = 14
AVG_LEN_TOKENS = 15
NUM_SENTENCES = 16
AVG_AOA = 17
AVG_IMG = 18
AVG_FAM = 19
STD_DEV_AOA = 20
STD_DEV_IMG = 21
STD_DEV_FAM = 22
AVG_V_MEAN_SUM = 23
AVG_A_MEAN_SUM = 24
AVG_D_MEAN_SUM = 25
STD_DEV_V_MEAN_SUM = 26
STD_DEV_A_MEAN_SUM = 27
STD_DEV_D_MEAN_SUM = 28

# comment categories and their indices
CATEGORIES = {
    "Left": 0,
    "Center": 1,
    "Right": 2,
    "Alt": 3
}

# global dicts to be populated in main
W_WORDS = {}
BGL_WORDS = {}
LIWCR_FEATS = {}
IDs = {
    "Left": {},
    "Center": {},
    "Right": {},
    "Alt": {}
}


def loadBGLWords():
    ''' This function loads words and their relevant props
        from the Bristol, Gilhooly and Logie norms csv file into a 
        global dict
    '''
    global BGL_WORDS

    root_dir = "/u/cs401/"

    with open(root_dir + "Wordlists/BristolNorms+GilhoolyLogie.csv", newline='') as csvfile:
        BGLReader = csv.reader(csvfile)
        next(BGLReader)

        for row in BGLReader:
            # error checking for empty rows
            if not row[1] or not row[3] or not row[4] or not row[5]:
                continue

            BGL_WORDS[row[1]] = {
                "AOA": float(row[3]),
                "IMG": float(row[4]),
                "FAM": float(row[5])
            }


def loadWWords():
    ''' This function loads words and their relevant props
        from the Warringer norms csv file into a global dict
    '''
    global W_WORDS

    root_dir = "/u/cs401/"

    with open(root_dir+'Wordlists/Ratings_Warriner_et_al.csv', newline='') as csvfile:
        WReader = csv.reader(csvfile)
        next(WReader)

        for row in WReader:
            # error checking for empty rows
            if not row[1] or not row[2] or not row[5] or not row[8]:
                continue

            W_WORDS[row[1]] = {
                "V_MEAN_SUM": float(row[2]),
                "A_MEAN_SUM": float(row[5]),
                "D_MEAN_SUM": float(row[8])
            }


def loadIDsAndLIWCFeats(a1_dir):
    ''' This function loads comment IDs and LIWC/Receptiviti features
        by category into global dicts

    Parameters:
        a1_dir : directory of A1
    '''
    global LIWCR_FEATS
    global IDs

    for cat in CATEGORIES:
        with open(os.path.join(a1_dir, 'feats/'+cat+'_IDs.txt'), "r") as f:
            for i, line in enumerate(f):
                line = re.sub(r'\s+', "", line)
                IDs[cat][line] = i

        LIWCR_FEATS[cat] = np.load(os.path.join(
            a1_dir, 'feats/'+cat+'_feats.dat.npy'))


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    feats = np.zeros(173)

    # arrays to store values to later calculate means and std deviations
    numTokensPerSent = []
    numCharsPerToken = []
    AOA = []
    IMG = []
    FAM = []
    V = []
    A = []
    D = []

    sentences = comment.split("\n")
    # remove last empty sentence
    del sentences[-1]

    for sent in sentences:
        tokens = sent.split(" ")
        numTokensPerSent.append(len(tokens))

        # keep track of whether will/shall is seen since it not appear
        # immediately before a VB token
        willSeen = False

        for i, tok in enumerate(tokens):
            # if tok is missing the tag or the lemma, skip to next tok
            try:
                token, tag = tok.rsplit('/', 1)
            except:
                continue

            # Extract feature 1 that relies on capitalization.
            if token.isupper() and len(token) >= 3:
                feats[UPPERCASE] += 1

            # Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
            token = token.lower()

            # keep track of whether will/shall is seen
            if token in {"will", "shall"}:
                willSeen = True

            # append to mean/std dev arrays
            if tag not in PUNCTUATION_TAGS:
                numCharsPerToken.append(len(token))

            if token in BGL_WORDS:
                AOA.append(BGL_WORDS[token]["AOA"])
                IMG.append(BGL_WORDS[token]["IMG"])
                FAM.append(BGL_WORDS[token]["FAM"])

            if token in W_WORDS:
                V.append(W_WORDS[token]["V_MEAN_SUM"])
                A.append(W_WORDS[token]["A_MEAN_SUM"])
                D.append(W_WORDS[token]["D_MEAN_SUM"])

            # Extract features (2-14) that do not rely on capitalization.
            if token in FIRST_PERSON_PRONOUNS:
                feats[FIRST_PERSON_PRONOUN] += 1

            elif token in SECOND_PERSON_PRONOUNS:
                feats[SECOND_PERSON_PRONOUN] += 1

            elif token in THIRD_PERSON_PRONOUNS:
                feats[THIRD_PERSON_PRONOUN] += 1

            if token in SLANG:
                feats[SLANG_ACRONYM] += 1

            if tag == "CC":
                feats[COORDINATING_CONJUCTION] += 1

            elif tag == "VBD":
                feats[PAST_TENSE_VERB] += 1

            elif tag == "VB":
                # "gonna VB"/"going to VB" case
                if i > 1 and tokens[i-1].split("/")[0].lower() == "to" and ((len(tokens[i-2].split("/")) > 1 and tokens[i-2].split("/")[1] == "VBG") or tokens[i-2].split("/")[0].lower() == "going"):
                    feats[FUTURE_TENSE_VERB] += 1

                elif willSeen:
                    feats[FUTURE_TENSE_VERB] += 1
                    willSeen = False

            elif tag == "," and len(token) == 1:
                feats[COMMA] += 1

            elif len(token) > 1 and tag in PUNCTUATION_TAGS:
                feats[MULTI_CHAR_PUNCTUATION] += 1

            elif tag in {"NN", "NNS"}:
                feats[COMMON_NOUN] += 1

            elif tag in {"NNP", "NNPS"}:
                feats[PROPER_NOUN] += 1

            elif tag in {"RB", "RBR", "RBS"}:
                feats[ADVERB] += 1

            elif tag in {"WDT", "WP", "WP$" "WRB"}:
                feats[WH_WORD] += 1

    # Extract features (15-17) by calculting mean/length
    if numCharsPerToken:
        feats[AVG_LEN_SENTENCES] = np.mean(numTokensPerSent)
    if numCharsPerToken:
        feats[AVG_LEN_TOKENS] = np.mean(numCharsPerToken)
    if sentences:
        feats[NUM_SENTENCES] = len(sentences)

    # Extract features (18-29) by calculting mean/std dev
    if AOA:
        feats[AVG_AOA] = np.mean(AOA)
        feats[STD_DEV_AOA] = np.std(AOA)
    if IMG:
        feats[AVG_IMG] = np.mean(IMG)
        feats[STD_DEV_IMG] = np.std(IMG)
    if FAM:
        feats[AVG_FAM] = np.mean(FAM)
        feats[STD_DEV_FAM] = np.std(FAM)

    if V:
        feats[AVG_V_MEAN_SUM] = np.mean(V)
        feats[STD_DEV_V_MEAN_SUM] = np.std(V)
    if A:
        feats[AVG_A_MEAN_SUM] = np.mean(A)
        feats[STD_DEV_A_MEAN_SUM] = np.std(A)
    if D:
        feats[AVG_D_MEAN_SUM] = np.mean(D)
        feats[STD_DEV_D_MEAN_SUM] = np.std(D)

    return feats


def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    # get LIWC/Receptiviti feats from the loaded dicts
    feats[29:173] = LIWCR_FEATS[comment_class][IDs[comment_class][comment_id]]
    return feats


def main(args):
    global IDs
    global BGL_WORDS
    global W_WORDS
    global LIWCR_FEATS

    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    a1_dir = args.a1_dir

    loadBGLWords()
    loadWWords()
    loadIDsAndLIWCFeats(a1_dir)

    for i in range(len(data)):
        comm = data[i]

        # Use extract1 to find the first 29 features for each
        # data point. Add these to feats.
        feats1 = extract1(comm["body"])

        # Use extract2 to copy LIWC/Receptiviti features (features 30-173)
        # into feats.
        comment_class = comm["cat"]
        feats2 = extract2(feats1, comment_class, comm["id"])

        feats[i][:173] = feats2
        feats[i][173] = CATEGORIES[comment_class]

        if i > 0 and i % 1000 == 0:
            print(f"Processed {i} comments")

    print(f"✓ Completed\n")
    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument(
        "-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument(
        "-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument(
        "-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)
