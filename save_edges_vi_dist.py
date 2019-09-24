
# find 30 nearest neighbors for each word in glove vector
import argparse
import logging
import re
from annoy import AnnoyIndex

def parse_args():
    parser = argparse.ArgumentParser()

    #iput glove vector file
    parser.add_argument(
        '--input_vectors', default = "glove.6B/glove.6B.50d.txt",
        help=
        'Unpack your data from https://nlp.stanford.edu/projects/glove/.\n'
        'Make sure to update DIMENSIONS according to embedded vector '
        'dimension')
    #Outputfile 
    parser.add_argument(
        '--out_file', default="edges.txt",
        help=
        'This file will contain nearest neighbors, one per line:\n'
        'node [tab char] neighbor_1 neighbor_2 ...')
    parser.add_argument(
        '--dimensions', type=int, default=50,
        help='How many dimension in the vector space')
    parser.add_argument(
        '--max_trees', type=int, default=50,
        help='How many trees do want to use for `AnnoyIndex`')
    return parser.parse_args()

if __name__ == '__main__':
    parameters = parse_args()
    with open(parameters.input_vectors) as input_file:
        lines = input_file.readlines()

    word_id = 0
    word_list = []
    word_index = AnnoyIndex(parameters.dimensions)

    for line in lines:
        line_array = line.split(' ')
        word = line_array[0]
        if re.search('[0-9\W]',word):
            continue
        word_list.append(word)
        vectors = [float(x) for x in line_array[1:]]
        word_index.add_item(word_id, vectors)
        word_id += 1

    word_index.build(parameters.max_trees)

    logging.info('Writing edges and distence in to {}..'.format(parameters.out_file))
    with open(parameters.out_file, 'w') as out:
        for i in range(len(word_list)):
            word = word_list[i]
            result = word_index.get_nns_by_item(i,30,include_distances=True)
            edges = [word_list[x] for x in result[0]]
            distences = result[1]
            distences = list(map(str, distences))
            pair_list = [None]*(len(edges)+len(distences))
            pair_list[::2] = edges
            pair_list[1::2] = distences
            if len(edges) > 0:
                out.write(word + '\t' + " ".join(pair_list)+'\n')
            if i % 10000 == 0:
                print(i)

    print("finished!")


#edges = [word_list[x] for x in result[0] if (word_list[x] != word)]
