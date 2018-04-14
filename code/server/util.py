from math import ceil
from time import time

import os


def chunkise(l, n):
    """
    Yield successive chunks of data of size n, and the rest of the data
    Args:
        l: list to chunkise,
        n: size of chunks
    Yeilds:
        list: chunk of size n
        list: Rest of data
    """
    for i in range(0, len(l), n):
        yield l[i:i + n], l[:i] + l[i+n:]


def list_chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    Yields:
        a list of length n
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def list_split(l, split_1, split_2):
    """
    Return two lists of l containing the values as indexes in split_1, split_2
    """
    ret_1 = [l[x] for x in split_1]
    ret_2 = [l[x] for x in split_2]

    return ret_1, ret_2


def flatten_list(l):
    """
    Flatten the list l by one level
    """
    return [item for sublist in l for item in sublist]


def train_val_test_split_idx(l):
    """
    Return 3 lists of l containing 80%, 10%, and 10% of the values for the training,
    validation and test set respectively
    """
    # Split into 10 equal chunks
    l_chunks = list()
    chunk_size = int(ceil(len(l) / float(10)))
    for chunk, _ in chunkise(l, chunk_size):
        l_chunks.append(chunk)

    return flatten_list(l_chunks[:8]), flatten_list(l_chunks[8:9]), l_chunks[-1]


def current_milli_time():
    return int(round(time() * 1000))


def time_function(function, *args):
    """
    Time a function, return time followed by function return values
    """

    start_time = current_milli_time()
    return_val = function(*args)
    end_time = current_milli_time()

    return (end_time - start_time, return_val)


def create_dir_if_not_exists(directory):
    """
    Create directory if it doesn't already exist
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
def calculate_percentage(variable, total):
    return 100*variable/len(total)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def get_img_num(img_filename):
    """
    parses the img_num from img_filename
    """
    img_num = img_filename[15:-5]

    if not is_number(img_num):
        img_num = img_filename[17:-5]

    return int(img_num)