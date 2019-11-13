import os
import codecs
import json
import pickle
import linearcorex as lc
import pandas as pd
from datetime import datetime

# pip install linearcorex numpy pandas scipy
# removed numpy max in transform() function of linearcorex
# removed order of TCs from linearcorex

INPUT_FILE = 'in/top3kvariance_plus_literature_genesymbol_transposed_for_corex.txt'
START_NUMBER_FACTORS = 900
MULTIPLIER_NUMBER_FACTORS = 2
END_NUMBER_FACTORS = 900
REPETITIONS = 1

OUTPUT_PARENT_DIRECTORY = 'out/'
OUTPUT_DIRECTORY = str(datetime.timestamp(datetime.now()))
OUTPUT_PATH = os.path.join(OUTPUT_PARENT_DIRECTORY, OUTPUT_DIRECTORY)

os.mkdir(OUTPUT_PATH)

input_matrix = pd.read_csv(INPUT_FILE, sep=' ', header=[0])
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
        fit = out.fit(input_matrix)
        transform = out.transform(input_matrix, details=True)
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
            best['fit'] = fit
            best['transform'] = transform
            print('BEST OF ALL REPETITIONS!')

        if repetition < REPETITIONS - 1:
            print('-----------------')

    print('========= RESULTS OF L.F. %d =========' % latent_factors)
    print('Clusters:\n%s' % str(best['clusters']))
    print('TCS:\n%s' % str(best['tcs']))
    print('TC:\n%.4f' % best['tc'])

    with codecs.open(os.path.join(OUTPUT_PATH, '%d.json' % latent_factors), 'w', encoding='utf-8') as json_file:
        json.dump({
            'clusters': best['clusters'].tolist(),
            'tcs': best['tcs'].tolist(),
            'tc': best['tc'],
        }, json_file, separators=(',', ':'), indent=2)

    with open(os.path.join(OUTPUT_PATH, 'fit_%d.pickle' % latent_factors), 'wb') as pickle_file:
        pickle.dump(best['fit'], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(OUTPUT_PATH, 'transform_%d.pickle' % latent_factors), 'wb') as pickle_file:
        pickle.dump(best['transform'], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    latent_factors *= 2
