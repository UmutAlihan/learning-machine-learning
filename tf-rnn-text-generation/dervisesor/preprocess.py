#!/usr/bin/env python3

import nltk, re, pprint


def scrub_words(text):
    """Basic cleaning of texts."""

    # remove html markup
    text=re.sub("(<.*?>)","",text)

    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)

    #remove whitespace
    text=text.strip()
    return text


with open ("cs-kitaplar.txt", "r") as f:
	data=f.readlines()

cleaned_words=[scrub_words(w) for w in raw_words]
