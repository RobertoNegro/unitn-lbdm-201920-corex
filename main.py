import os
import codecs
import json
import pickle
import linearcorex as lc
import pandas as pd
from datetime import datetime

# pip install linearcorex numpy pandas scipy

INPUT_FILE = 'in/top3k_genesymbol_variance_transposed_for_corex.txt'
START_NUMBER_FACTORS = 640
MULTIPLIER_NUMBER_FACTORS = 2
END_NUMBER_FACTORS = 5120
REPETITIONS = 1

OUTPUT_PARENT_DIRECTORY = 'out/'
OUTPUT_DIRECTORY = str(datetime.timestamp(datetime.now()))
OUTPUT_PATH = os.path.join(OUTPUT_PARENT_DIRECTORY, OUTPUT_DIRECTORY)

os.mkdir(OUTPUT_PATH)

input_matrix = pd.read_csv(INPUT_FILE, sep=' ', header=[0], index_col=0)
print('Using as input matrix:\n%s' % str(input_matrix))

latent_factors = START_NUMBER_FACTORS
while latent_factors <= END_NUMBER_FACTORS:
    print('========= LATENT FACTOR %d =========' % latent_factors)

    best = {
        'clusters': [],
        'tcs': [],
        'tc': 0,
    }
    for repetition in range(0, REPETITIONS):
        print('Executing with %d latent factors, repetition %d...' % (latent_factors, repetition))
        print('Fitting...')
        out = lc.Corex(n_hidden=latent_factors, max_iter=10000, verbose=True)
        out.fit(input_matrix)
        Y = out.transform(input_matrix, details=True)

        clusters = out.clusters()
        tcs = out.tcs
        tc = out.tc

        print('Clusters:\n%s' % str(clusters))
        print('TCS:\n%s' % str(tcs))
        print('TC:\n%.4f' % tc)

        if tc > best['tc']:
            best['clusters'] = clusters
            best['tcs'] = tcs
            best['tc'] = tc
            best['Y'] = Y
            print('BEST OF ALL REPETITIONS!')

        if repetition < REPETITIONS - 1:
            print('-----------------')

    print('========= RESULTS OF L.F. %d =========' % latent_factors)
    print('Clusters:\n%s' % str(best['clusters']))
    print('TCS:\n%s' % str(best['tcs']))
    print('TC:\n%.4f' % best['tc'])

    with codecs.open(os.path.join(OUTPUT_PATH, '%d.csv' % latent_factors), 'w', encoding='utf-8') as json_file:
        json.dump({
            'clusters': best['clusters'].tolist(),
            'tcs': best['tcs'].tolist(),
            'tc': best['tc'],
        }, json_file, separators=(',', ':'), indent=2)

    with open(os.path.join(OUTPUT_PATH, '%d.pickle' % latent_factors), 'wb') as pickle_file:
        pickle.dump(best['Y'], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    latent_factors *= 2
