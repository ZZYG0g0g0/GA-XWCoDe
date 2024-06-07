from collections import Counter
import pandas as pd
import Set_generation


def UniqueWordCount(fname, tname):
    queried_line = Set_generation.set_generation(fname)
    query_line = Set_generation.set_generation(tname)
    df = pd.DataFrame(index=range(len(query_line)), columns=range(len(queried_line)))
    for i in range(len(query_line)):
        for j in range(len(queried_line)):
            c = 0
            word_counts = Counter(queried_line[j])
            for word, count in word_counts.items():
                if count == 1:
                    c += 1
            df.iloc[i][j] = c
    return df


def TotalWordCount(fname, tname):
    queried_line = Set_generation.set_generation(fname)
    query_line = Set_generation.set_generation(tname)
    df = pd.DataFrame(index=range(len(query_line)), columns=range(len(queried_line)))
    for i in range(len(query_line)):
        for j in range(len(queried_line)):
            df.iloc[i][j] = len(queried_line[j])
    return df


def PairOverlap(fname, tname):
    percentageOverlap = 0
    queried_line = Set_generation.set_generation(fname)
    query_line = Set_generation.set_generation(tname)
    df = pd.DataFrame(index=range(len(query_line)), columns=range(len(queried_line)))
    for i in range(len(query_line)):
        for j in range(len(queried_line)):
            set1 = set(query_line[i])
            set2 = set(queried_line[j])
            overlap = set1 & set2
            union = set1 | set2
            percentageOverlap = float(len(overlap)) / len(union) * 100
            df.iloc[i][j] = percentageOverlap
    return df


if __name__ == "__main__":
    query_file = "./UC_clear.txt"
    queried_file = "./CN_MN_VN_CMT_clear.txt"
    queried_line = Set_generation.set_generation(queried_file)
    query_line = Set_generation.set_generation(query_file)
    print(UniqueWordCount(queried_line, query_line))
    print(TotalWordCount(queried_line, query_line))
    print(PairOverlap(queried_line, query_line))
