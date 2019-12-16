import os
import pandas as pd
from datetime import datetime


def initializeCluster(input, ids):
    cluster = pd.DataFrame(input.head())
    for i, el in ids.iterrows():
        id = el[0]
        cluster.loc[id] = input.loc[id, :]
    return cluster


def getMedians(cluster):
    medians = pd.DataFrame(columns=['median'])
    for id, el in cluster.iteritems():
        medians.loc[id] = cluster[id].median()
    return medians


# ------------- CONFIG ----------------
INPUT_FILE = 'in/top3kvariance_plus_literature_genesymbol_transposed_for_corex_with_sample.txt'
INPUT_FILE_CLUSTER_1 = 'in/cluster1.txt'
INPUT_FILE_CLUSTER_2 = 'in/cluster2.txt'
INPUT_FILE_CLUSTER_3 = 'in/cluster3.txt'

OUTPUT_PARENT_DIRECTORY = 'out/'
OUTPUT_DIRECTORY = str(datetime.timestamp(datetime.now()))
OUTPUT_PATH = os.path.join(OUTPUT_PARENT_DIRECTORY, OUTPUT_DIRECTORY)
# -------------------------------------

input = pd.read_csv(INPUT_FILE, index_col=3004, delimiter=r"\s+")

cluster1_ids = pd.read_csv(INPUT_FILE_CLUSTER_1, header=None, delimiter=r"\s+")
cluster2_ids = pd.read_csv(INPUT_FILE_CLUSTER_2, header=None, delimiter=r"\s+")
cluster3_ids = pd.read_csv(INPUT_FILE_CLUSTER_3, header=None, delimiter=r"\s+")

cluster1 = initializeCluster(input, cluster1_ids)
cluster2 = initializeCluster(input, cluster2_ids)
cluster3 = initializeCluster(input, cluster3_ids)

cluster1_2 = pd.concat([cluster1, cluster2])
cluster1_3 = pd.concat([cluster1, cluster3])
cluster2_3 = pd.concat([cluster2, cluster3])

medians1 = getMedians(cluster1)
medians2 = getMedians(cluster2)
medians3 = getMedians(cluster3)

medians1_2 = getMedians(cluster1_2)
medians1_3 = getMedians(cluster1_3)
medians2_3 = getMedians(cluster2_3)

result1 = medians1 - medians2_3
result2 = medians2 - medians1_3
result3 = medians3 - medians1_2

result1_sorted = result1.sort_values(by=['median'])
result2_sorted = result2.sort_values(by=['median'])
result3_sorted = result3.sort_values(by=['median'])

os.mkdir(OUTPUT_PATH)

medians1.to_csv(os.path.join(OUTPUT_PATH, 'medians1.csv'), sep=',')
medians2_3.to_csv(os.path.join(OUTPUT_PATH, 'medians2_3.csv'), sep=',')
result1.to_csv(os.path.join(OUTPUT_PATH, 'diff1.csv'), sep=',')
result2.to_csv(os.path.join(OUTPUT_PATH, 'diff2.csv'), sep=',')
result3.to_csv(os.path.join(OUTPUT_PATH, 'diff3.csv'), sep=',')
result1_sorted.to_csv(os.path.join(OUTPUT_PATH, 'diff1_sorted.csv'), sep=',')
result2_sorted.to_csv(os.path.join(OUTPUT_PATH, 'diff2_sorted.csv'), sep=',')
result3_sorted.to_csv(os.path.join(OUTPUT_PATH, 'diff3_sorted.csv'), sep=',')