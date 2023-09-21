#
# This file is part of the SentiCR Project (https://github.com/senticr/SentiCR).
#
#  Ahmed, T. , Bosu, A., Iqbal, A. and Rahimi, S., "SentiCR: A Customized
#  Sentiment Analysis Tool for Code Review Interactions", In Proceedings of the
#  32nd IEEE/ACM International Conference on Automated Software Engineering
#  (NIER track).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# Modified by happygirlzt

# Justin says: I've diced this up a lot, swapped a few things for Python 3.7,
# and it runs different than the the paper version, so if you think this is
# copy-paste, you'd be mistaken.
#
# Buyer beware. There be dragons.
__author__ = "Justin Ribeiro <justin@justinribeiro.com>"

from datetime import datetime
import logging
from pickle import TRUE
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

import time
import random
import csv
import re

import nltk
from statistics import mean

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import SMOTE

from joblib import dump, load

from pathlib import Path

data_folder = Path("/work/src/SentiCR/SentiCR/data/PTM/")
save_folder = Path("/work/src/SentiCR/SentiCR/")

gh_train = data_folder / "gh-train.pkl"
gh_test = data_folder / "gh-test.pkl"


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


stemmer = SnowballStemmer("english")


def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems


mystop_words = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ourselves",
    "you",
    "your",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "themselves",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "if",
    "or",
    "as",
    "until",
    "of",
    "at",
    "by",
    "between",
    "into",
    "through",
    "during",
    "to",
    "from",
    "in",
    "out",
    "on",
    "off",
    "then",
    "once",
    "here",
    "there",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "other",
    "some",
    "such",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "don",
    "should",
    "now",
    # keywords
    "while",
    "case",
    "switch",
    "def",
    "abstract",
    "byte",
    "continue",
    "native",
    "private",
    "synchronized",
    "if",
    "do",
    "include",
    "each",
    "than",
    "finally",
    "class",
    "double",
    "float",
    "int",
    "else",
    "instanceof",
    "long",
    "super",
    "import",
    "short",
    "default",
    "catch",
    "try",
    "new",
    "final",
    "extends",
    "implements",
    "public",
    "protected",
    "static",
    "this",
    "return",
    "char",
    "const",
    "break",
    "boolean",
    "bool",
    "package",
    "byte",
    "assert",
    "raise",
    "global",
    "with",
    "or",
    "yield",
    "in",
    "out",
    "except",
    "and",
    "enum",
    "signed",
    "void",
    "virtual",
    "union",
    "goto",
    "var",
    "function",
    "require",
    "print",
    "echo",
    "foreach",
    "elseif",
    "namespace",
    "delegate",
    "event",
    "override",
    "struct",
    "readonly",
    "explicit",
    "interface",
    "get",
    "set",
    "elif",
    "for",
    "throw",
    "throws",
    "lambda",
    "endfor",
    "endforeach",
    "endif",
    "endwhile",
    "clone",
    "ani",
    "continu",
    "deleg",
    "doe",
    "doubl",
    "dure",
    "els",
    "endwhil",
    "extend",
    "implement",
    "includ",
    "interfac",
    "namespac",
    "nativ",
    "onc",
    "ourselv",
    "overrid",
    "packag",
    "privat",
    "protect",
    "rais",
    "readon",
    "requir",
    "sign",
    "synchron",
    "themselv",
    "tri",
    "veri",
    "yourselv",
    "el",
    "rai",
]

logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s")


emodict = []
contractions_dict = []


# Read in the words with sentiment from the dictionary
with open("Contractions.txt", "r") as contractions, open(
    "EmoticonLookupTable.txt", "r"
) as emotable:
    contractions_reader = csv.reader(contractions, delimiter="\t")
    emoticon_reader = csv.reader(emotable, delimiter="\t")

    # Hash words from dictionary with their values
    contractions_dict = {rows[0]: rows[1] for rows in contractions_reader}
    emodict = {rows[0]: rows[1] for rows in emoticon_reader}

    contractions.close()
    emotable.close()

grammar = r"""
NegP: {<VERB>?<ADV>+<VERB|ADJ>?<PRT|ADV><VERB>}
{<VERB>?<ADV>+<VERB|ADJ>*<ADP|DET>?<ADJ>?<NOUN>?<ADV>?}

"""
chunk_parser = nltk.RegexpParser(grammar)


contractions_regex = re.compile("(%s)" % "|".join(contractions_dict.keys()))


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_regex.sub(replace, str(s).lower())


url_regex = re.compile(
    "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def remove_url(s):
    return url_regex.sub(" ", s)


negation_words = [
    "not",
    "never",
    "none",
    "nobody",
    "nowhere",
    "neither",
    "barely",
    "hardly",
    "nothing",
    "rarely",
    "seldom",
    "despite",
]

emoticon_words = ["PositiveSentiment", "NegativeSentiment"]


def negated(input_words):
    """
    Determine if input contains negation words
    """
    neg_words = []
    neg_words.extend(negation_words)
    for word in neg_words:
        if word in input_words:
            return True
    return False


def prepend_not(word):
    if word in emoticon_words:
        return word
    elif word in negation_words:
        return word
    return "NOT_" + word


def handle_negation(comments):
    sentences = nltk.sent_tokenize(comments)
    modified_st = []
    for st in sentences:
        allwords = nltk.word_tokenize(st)
        modified_words = []
        if negated(allwords):
            part_of_speech = nltk.tag.pos_tag(allwords, tagset="universal")
            chunked = chunk_parser.parse(part_of_speech)
            # print("---------------------------")
            # print(st)
            for n in chunked:
                if isinstance(n, nltk.tree.Tree):
                    words = [pair[0] for pair in n.leaves()]
                    # print(words)

                    if n.label() == "NegP" and negated(words):
                        for i, (word, pos) in enumerate(n.leaves()):
                            if (pos == "ADV" or pos == "ADJ" or pos == "VERB") and (
                                word != "not"
                            ):
                                modified_words.append(prepend_not(word))
                            else:
                                modified_words.append(word)
                    else:
                        modified_words.extend(words)
                else:
                    modified_words.append(n[0])
            newst = " ".join(modified_words)
            # print(newst)
            modified_st.append(newst)
        else:
            modified_st.append(st)
    return ". ".join(modified_st)


def preprocess_text(text):
    text = str(text)
    comments = text.encode("ascii", "ignore").decode("utf-8")
    comments = expand_contractions(comments)
    comments = remove_url(comments)
    comments = replace_all(comments, emodict)
    comments = handle_negation(comments)

    return comments


class SentimentData:
    def __init__(self, text, rating):
        self.text = text
        self.rating = rating


class SentiCR:
    # Change the training oracle
    def __init__(
        self, algo="GBT", save_model=False, pretrained_model=None, training_data=None
    ):
        print(
            "🚀 STARTING ENGINES: Justin's Creative Sentiment Scoring Fork...Burn Baby Burn 🔥"
        )
        self.algo = algo
        if training_data is None:
            self.training_data = self.read_data_from_oracle_gh()
        else:
            self.training_data = training_data
        begin = time.time()
        self.model = self.create_model_from_training_data(save_model, pretrained_model)
        end = time.time()
        print("Training used {:.2f} seconds".format(end - begin))

    def read_data_from_oracle_gh(self):
        oracle_data = []
        cur_train = pd.read_pickle(gh_train)
        print("READ: Github PR training set...", end=" ")

        for index, row in cur_train.iterrows():
            oracle_data.append(SentimentData(row["sentence"], row["label"]))
        print("DONE")

        return oracle_data

    def get_classifier(self):
        algo = self.algo

        if algo == "GBT":
            return GradientBoostingClassifier()
        elif algo == "RF":
            return RandomForestClassifier()
        elif algo == "ADB":
            return AdaBoostClassifier()
        elif algo == "DT":
            return DecisionTreeClassifier()
        elif algo == "NB":
            return BernoulliNB()
        elif algo == "SGD":
            return SGDClassifier()
        elif algo == "SVC":
            return LinearSVC()
        elif algo == "MLPC":
            return MLPClassifier(
                activation="logistic",
                batch_size="auto",
                early_stopping=True,
                hidden_layer_sizes=(100,),
                learning_rate="adaptive",
                learning_rate_init=0.1,
                max_iter=5000,
                random_state=1,
                solver="lbfgs",
                tol=0.0001,
                validation_fraction=0.1,
                verbose=False,
                warm_start=False,
            )
        return 0

    def create_model_from_training_data(self, save_model=False, pretrained_model=None):
        training_comments = []
        training_ratings = []
        print("READY: Training classifier model.....")
        print("Pre-processing PR Review Set text and ratings...", end=" ")
        for sentidata in self.training_data:
            comments = preprocess_text(sentidata.text)
            training_comments.append(comments)
            training_ratings.append(sentidata.rating)
        print("DONE")

        # discard stopwords, apply stemming, and discard words present in less than 3 comments
        print("Defining Vectorizer...", end=" ")
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            sublinear_tf=True,
            max_df=0.5,
            stop_words=mystop_words,
            min_df=3,
        )
        print("DONE")

        print("Running Fit Transform...", end=" ")
        X_train = self.vectorizer.fit_transform(training_comments).toarray()
        Y_train = np.array(training_ratings)
        print("DONE")

        # Apply SMOTE to improve ratio of the minority class
        print("Applying SMOTE to Improve ratio...", end=" ")
        smote_model = SMOTE(sampling_strategy="auto", k_neighbors=15)
        print("DONE")

        # smote_model = SMOTE(random_state=2)
        print("Applying SMOTE.fit_resample...", end=" ")
        X_resampled, Y_resampled = smote_model.fit_resample(X_train, Y_train)
        print("DONE")

        if pretrained_model is not None:
            filepath = save_folder / pretrained_model
            model = load(filename=filepath)
            print("Loaded existing model...DONE", flush=True)
        else:
            print("Running model fit...", end=" ", flush=True)
            model = self.get_classifier()
            model.fit(X_resampled, Y_resampled)
            print("DONE")

        self.model = model

        (precision, recall, f1score, accuracy) = self.ten_fold_cross_validation()

        if save_model is True:
            filename = "gh-trainingset-pr-code-reviews"
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d-%H:%M:%S")
            filename = save_folder / f"{filename}-{date_time}.joblib"
            dump(model, filename)
            print(f"Saved Model: {filename}")

        return model

    def get_sentiment_polarity(self, text):
        comment = preprocess_text(text)
        feature_vector = self.vectorizer.transform([comment]).toarray()
        sentiment_class = self.model.predict(feature_vector)
        return sentiment_class

    def get_sentiment_polarity_collection(self, texts):
        predictions = []
        for text in texts:
            comment = preprocess_text(text)
            feature_vector = self.vectorizer.transform([comment]).toarray()
            sentiment_class = self.model.predict(feature_vector)
            predictions.append(sentiment_class)

        return predictions

    def ten_fold_cross_validation(self):
        kf = KFold(n_splits=10)

        run_precision = []
        run_recall = []
        run_f1score = []
        run_accuracy = []

        count = 1
        print("Running 10-Fold Cross Validation....")

        dataset = np.array(self.read_data_from_oracle_gh())
        # Randomly divide the dataset into 10 partitions
        # During each iteration one partition is used for test and remaining 9 are used for training
        for train, test in kf.split(dataset):
            print("⚡ Using split-" + str(count) + " as test data set...", end=" ")

            test_comments = [comments.text for comments in dataset[test]]
            test_ratings = [comments.rating for comments in dataset[test]]

            pred = self.get_sentiment_polarity_collection(test_comments)

            precision = precision_score(test_ratings, pred, average="weighted")
            recall = recall_score(test_ratings, pred, average="weighted")
            f1score = f1_score(test_ratings, pred, average="weighted")
            accuracy = accuracy_score(test_ratings, pred)

            run_accuracy.append(accuracy)
            run_f1score.append(f1score)
            run_precision.append(precision)
            run_recall.append(recall)
            count += 1
            print("DONE")
        print(" ")
        print("Validation Results for Model")
        print("-----------------------------------")
        print("Precision: " + str(precision))
        print("Recall:    " + str(recall))
        print("F-measure: " + str(f1score))
        print("Accuracy:  " + str(accuracy))
        print("-----------------------------------")
        print(" ")

        return (
            mean(run_precision),
            mean(run_recall),
            mean(run_f1score),
            mean(run_accuracy),
        )
