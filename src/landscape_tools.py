import numpy as np, src.sparray as sp
from scipy.sparse import dok_matrix


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

def dH(s1, s2):
    # compute Hamming distance between two genotype sequences
    return sum([1 for a, b in zip(s1, s2) if a!=b])

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

def locmax(q, L, fitness, avseqs=[], nrmax=np.inf):
    # find local maxima of a landscape in sequence space
        
    fmat = np.nan*np.ones([q]*L, dtype='float')
    if len(avseqs) == 0:
        for seq in seqlist(q=q, L=L):
            fmat[seq] = fitness(seq=seq)
    else:
        for seq in avseqs:
            fmat[seq] = fitness(seq=seq)
        fmat[np.isnan(fmat)] = -np.inf

    # ls_dx:= dummy copies of the landscape, maxs:= dict with positions and values of local maxima in the landscape
    fmat_d1, fmat_d2, smax = fmat.copy(), fmat.copy(), {}

    while np.max(fmat_d1) > -np.inf and len(smax) < nrmax:
        # find position of maximal remaining fitness in landscape, and add it to dict
        inds = np.unravel_index(np.argmax(fmat_d2, axis=None), fmat.shape)
        if fmat_d1[inds] > -np.inf:
            smax[inds] = fmat[inds]
        # set local maximum to minus infinity
        fmat_d2[inds] = -np.inf
        # set all neighbors of that maximum (seqs accessible within one mutation) to minus infinity (as they cannot be local maxima)
        for i in range(L):
            fmat_d1[inds[:i]+(slice(fmat.shape[i]),)+inds[i+1:]] = -np.inf

    return smax

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

def mkM(q=2, L=10, order=2, ising=False, seqs=[]):
    # computes matrix M that maps vector of model parameters X to vector
    # of fitness values F: F=M.X
    if len(seqs) == 0:
        seqs = seqlist(q=q, L=L)
    
    if order == 1:
        M = dok_matrix((len(seqs), L), dtype=int)
        for x, seq in enumerate(seqs):
            for i in range(L):
                if ising:
                    M[x,i] = 2*seq[i]-1
                else:
                    if seq[i]==1:
                        M[x,i] = 1
    
    elif order == 2:
        M = dok_matrix((len(seqs), L+int(L*(L-1)/2)), dtype=int)
        for x, seq in enumerate(seqs):
            jcnt = 0
            for i in range(L):
                if ising:
                    M[x,i] = 2*seq[i]-1
                else:
                    if seq[i]==1:
                        M[x,i] = 1
                for j in range(i):
                    if ising:
                        M[x,L+jcnt] = (2*seq[i]-1)*(2*seq[j]-1)
                    else:
                        if seq[i]==1 and seq[j]==1:
                            M[x,L+jcnt] = 1
                    jcnt += 1
    
    elif order == 3:
        M = dok_matrix((len(seqs), L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))), dtype=int)
        for x, seq in enumerate(seqs):
            jcnt, kcnt = 0, 0
            for i in range(L):
                if ising:
                    M[x,i] = 2*seq[i]-1
                else:
                    if seq[i]==1:
                        M[x,i] = 1
                for j in range(i):
                    if ising:
                        M[x,L+jcnt] = (2*seq[i]-1)*(2*seq[j]-1)
                    else:
                        if seq[i]==1 and seq[j]==1:
                            M[x,L+jcnt] = 1
                    jcnt += 1
                    for k in range(j):
                        if ising:
                            M[x,L+int(L*(L-1)/2)+kcnt] = (2*seq[i]-1)*(2*seq[j]-1)*(2*seq[k]-1)
                        else:
                            if seq[i]==1 and seq[j]==1 and seq[k]==1:
                                M[x,L+int(L*(L-1)/2)+kcnt] = 1
                        kcnt += 1
    
    elif order == 4:
        M = dok_matrix((len(seqs), L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))+int(L*(L-1)*(L-2)*(L-3)/(4*3*2))), dtype=int)
        for x, seq in enumerate(seqs):
            jcnt, kcnt, hcnt = 0, 0, 0
            for i in range(L):
                if ising:
                    M[x,i] = 2*seq[i]-1
                else:
                    if seq[i]==1:
                        M[x,i] = 1
                for j in range(i):
                    if ising:
                        M[x,L+jcnt] = (2*seq[i]-1)*(2*seq[j]-1)
                    else:
                        if seq[i]==1 and seq[j]==1:
                            M[x,L+jcnt] = 1
                    jcnt += 1
                    for k in range(j):
                        if ising:
                            M[x,L+int(L*(L-1)/2)+kcnt] = (2*seq[i]-1)*(2*seq[j]-1)*(2*seq[k]-1)
                        else:
                            if seq[i]==1 and seq[j]==1 and seq[k]==1:
                                M[x,L+int(L*(L-1)/2)+kcnt] = 1
                        kcnt += 1
                        for h in range(k):
                            if ising:
                                M[x,L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))+hcnt] = (2*seq[i]-1)*(2*seq[j]-1)*(2*seq[k]-1)*(2*seq[h]-1)
                            else:
                                if seq[i]==1 and seq[j]==1 and seq[k]==1 and seq[h]==1:
                                    M[x,L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))+hcnt] = 1
                            hcnt += 1
    
    elif order == 5:
        M = dok_matrix((len(seqs), L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))+int(L*(L-1)*(L-2)*(L-3)/(4*3*2))+int(L*(L-1)*(L-2)*(L-3)*(L-4)/(5*4*3*2))), dtype=int)
        for x, seq in enumerate(seqs):
            jcnt, kcnt, hcnt, gcnt = 0, 0, 0, 0
            for i in range(L):
                if ising:
                    M[x,i] = 2*seq[i]-1
                else:
                    if seq[i]==1:
                        M[x,i] = 1
                for j in range(i):
                    if ising:
                        M[x,L+jcnt] = (2*seq[i]-1)*(2*seq[j]-1)
                    else:
                        if seq[i]==1 and seq[j]==1:
                            M[x,L+jcnt] = 1
                    jcnt += 1
                    for k in range(j):
                        if ising:
                            M[x,L+int(L*(L-1)/2)+kcnt] = (2*seq[i]-1)*(2*seq[j]-1)*(2*seq[k]-1)
                        else:
                            if seq[i]==1 and seq[j]==1 and seq[k]==1:
                                M[x,L+int(L*(L-1)/2)+kcnt] = 1
                        kcnt += 1
                        for h in range(k):
                            if ising:
                                M[x,L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))+hcnt] = (2*seq[i]-1)*(2*seq[j]-1)*(2*seq[k]-1)*(2*seq[h]-1)
                            else:
                                if seq[i]==1 and seq[j]==1 and seq[k]==1 and seq[h]==1:
                                    M[x,L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))+hcnt] = 1
                            hcnt += 1
                            for g in range(h):
                                if ising:
                                    M[x,L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))+int(L*(L-1)*(L-2)*(L-3)/(4*3*2))+gcnt] = (2*seq[i]-1)*(2*seq[j]-1)*(2*seq[k]-1)*(2*seq[h]-1)*(2*seq[g]-1)
                                else:
                                    if seq[i]==1 and seq[j]==1 and seq[k]==1 and seq[h]==1 and seq[g]==1:
                                        M[x,L+int(L*(L-1)/2)+int(L*(L-1)*(L-2)/(3*2))+int(L*(L-1)*(L-2)*(L-3)/(4*3*2))+gcnt] = 1
                                gcnt += 1
    return M