import numpy as np, src.sparray as sp


def seqlist(q=2, L=1):
    # enumerates at q**L possible sequences as tuples in a list
    seqs = [[0 for i in range(L)]]
    for m in range(q**L-1):
        seq = seqs[-1].copy()
        seq[-1] += 1
        for i in range(L)[::-1]:
            if seq[i] >= q:
                seq[i] = 0
                seq[i-1] += 1
        seqs.append(seq)
    return [tuple(seq) for seq in seqs]

class EmpLS:
    # a class for empirical landscapes

    def __init__(self, L=1, q=2, seqs=[], fs=[], default=np.nan):
        self.L, self.q = L, q
        self.seqs, self.fs = seqs, fs
        self.fmat = sp.sparray(shape=[self.q]*self.L, default=default, dtype=float)
        for seq, f in zip(seqs, fs):
            self.fmat[seq] = f
                
    def fitness(self, seq=(0)):
        return self.fmat[seq]

def mutate(seq, sites):
    # flip the state of the specified 'sites' in binary sequence 'seq'
    seq = list(seq)
    for i in sites:
        seq[i] = 1-seq[i]
    return tuple(seq)

def gammaij(L, seqs, fitness):
    # compute epistatic map gamma_i->j
    gamma = np.nan*np.ones((L, L))
    for i in range(L):
        for j in range(L):
            nom, denom = 0., 0.
            for s in seqs:
                if i != j:
                    nom += (fitness(mutate(s, [j])) - fitness(s)) * (fitness(mutate(s, [i,j])) - fitness(mutate(s, [i])))
                else:
                    nom += (fitness(mutate(s, [j])) - fitness(s)) * (fitness(mutate(s, [j])) - fitness(mutate(s, [])))
                denom += (fitness(mutate(s, [j])) - fitness(s))**2
            gamma[i,j] = nom/denom
    return gamma