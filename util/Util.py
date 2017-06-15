import logging
import re
from typing import List, Dict
import urllib
import urllib.request
from logging.handlers import RotatingFileHandler
import subprocess
import os

def get_logger(name: str, debug_log_file_name: str): # -> logging.Logger:
    alogger = logging.getLogger(name)
    alogger.setLevel(logging.DEBUG) # CAREFUL ==> need this, otherwise everybody chokes!
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s [%(module)s.%(funcName)s:%(lineno)d => %(message)s]')
    #
    create_debug_handler = False
    # fh = logging.FileHandler(debug_log_file_name)
    fh = RotatingFileHandler(debug_log_file_name, mode='a', maxBytes=5 * 1024 * 1024, backupCount=2, encoding=None, delay=0)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    if not len(alogger.handlers):
        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        # and add it to logger
        alogger.addHandler(ch)

        # we need a debug handler: let's flag our needs:
        create_debug_handler = True

        print("Logger created")
    else:
        print("Logger retrieved")
        # did the file handler change names?
        curr_debug_handler = alogger.handlers[1]
        if curr_debug_handler.baseFilename != fh.baseFilename:
            print("Changing log file names; was '{}', switching to '{}'".format(curr_debug_handler.baseFilename,
                                                                                fh.baseFilename))
            alogger.removeHandler(curr_debug_handler)
            # we need a debug handler: let's flag our needs:
            create_debug_handler = True
        else:
            # the debug handler we have is all good!
            create_debug_handler = False

    # If we need a debug handler, let's create it!
    if create_debug_handler:
        print("Creating debug handler at '{}'".format(fh.baseFilename))
        alogger.addHandler(fh)

    s = "'{}': logging 'INFO'+ logs to Console, 'DEBUG'+ logs to '{}'".format(alogger.name, alogger.handlers[1].baseFilename)
    print(s)
    alogger.info(s)
    alogger.debug(s)
    return alogger



# A custom function to clean the text before sending it into the vectorizer
def clean_text(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ").replace("&#039;", "'").replace("&quot;", "\"")

    # replace twitter @mentions
    #     mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
    #     text = mentionFinder.sub("@MENTION", text)

    # replace HTML symbols
    text = text.replace("&amp;", "and").replace("&gt;", ">").replace("&lt;", "<").replace("<br>", " ").replace("<BR>",
                                                                                                               " ")

    #     # lowercase
    #     text = text.lower()

    return text

emoji_replacements = {
    ":D": "laughter",
    "lol": "laughter",
    ":-)": "smile",
    ":)": "smile",
}

def replace_3_or_more_letters(s: str) -> str:
    """
        Replaces 3 or more consecutive letters by 2 of them.
        Motivation: there are no words in English that has 3 letters consecutive.  
    :param s: a word
    :return: the word with 3+ more repetitions replaced by 2 letters.
    """
    return re.sub(r'(.)\1+', r'\1\1', s)

def smart_split_in_words(a_sentence: str) -> List[str]:
    clean_sentence = a_sentence.replace('!', " ! ").replace('?', " ? ")
    for orig, repl in emoji_replacements.items():
        clean_sentence = clean_sentence.replace(orig, repl)
    return list(map(replace_3_or_more_letters, clean_sentence.split()))


def download_file(url, fname):
    urllib.request.urlretrieve(url, fname)

flatten = lambda l: [item for sublist in l for item in sublist]

def get_git_root() -> str:
    """
    Gets git root of a project. If the code is not on git, returns empty string.
    See http://stackoverflow.com/questions/22081209/find-the-root-of-the-git-repository-where-the-file-lives
    """
    try:
        return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip()\
            .decode("utf-8")  # conversion to string
    except:
        return ""


def append_to_git_root(what: str, alternate_root: str) -> str:
    """
    Appends a path to git root, or to an alternate path (if this code is not running 
    on a git-controlled environment)
    :param what: a path
    :param alternate_root: a directory where to append if git root is not defined
    :return: a path
    """
    git_root = get_git_root()
    if (git_root == ''):
        return os.path.join(alternate_root, what)
    else:
        return os.path.join(git_root, what)


