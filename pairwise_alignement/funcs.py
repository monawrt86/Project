'''
Created on Mar 6, 2024

@author: danhbuithi
'''
import numpy as np 
import utils
from scipy import stats

def context_score(x, i, j, L1, L2, k=1):
    #neighbors = []
    
    t = 1
    c = []
    while (t <= k and i - t >= 0 and j - t >= 0):
        c.append(x[i-t, j-t])
        t += 1 
    #if len(c) > 0:
    #    neighbors.append(np.sum(c)/len(c))
        
    t = 1
    #c = []
    if (t <= k and i+t < L1 and j+t < L2):
        c.append(x[i+t, j+t])
        t += 1
    #if len(c) > 0:
    #    neighbors.append(np.sum(c)/len(c))
    
    u = 0 
    if len(c) > 0: u = np.sum(c)/len(c)#max(neighbors)     
    return 0.6 * x[i, j] + 0.4 * u
    

def SW_compute_score(sim_matrix, gap_penalty):
    '''
    Compute Smith - Waterman score using dynamic programming algorithm
    '''
    L1, L2 = sim_matrix.shape
    
    H = np.zeros((L1+1, L2+1))
    
    T = np.zeros((L1+1, L2+1))
    
    for i in range(1, L1+1):
        for j in range(1, L2+1):
            options = []
            
            #a = H[i-1, j-1] + sim_matrix[i-1,j-1]
            a = H[i-1, j-1] + context_score(sim_matrix, i-1,j-1, L1, L2)
                
            options.append((a, 4)) #diag
            
            b = H[i, j-1] - gap_penalty
            options.append((b, 3)) #left
             
            c = H[i-1, j] - gap_penalty
            options.append((c, 2)) #up
             
            options.append((0.0, 1)) #nothing
            
            d = max(options)
            
            H[i, j] = d[0] 
            T[i, j] = d[1] 
                      
    return H, T  

def SW_traceback(L1, L2, H, T):
    '''
    Traceback for Smith - Waterman pairwise alignment algorithm
    '''
    
    v = np.argmax(H)
    i = v // (L2+1)
    j = v % (L2+1)
        
    aligned_seq1 = []
    aligned_seq2 = []
    
    ri = i 
    rj = j 
    
    while (H[i, j] > 0):
        if T[i, j] == 4: #diag
            aligned_seq1.insert(0, True)
            aligned_seq2.insert(0, True)
            i -= 1 
            j -= 1 
            
        elif T[i, j] == 3: #left 
            aligned_seq1.insert(0, False)
            aligned_seq2.insert(0, True)
            j -= 1 
        elif T[i, j] == 2: #up 
            aligned_seq1.insert(0, True)
            aligned_seq2.insert(0, False)
            i -= 1
           
    if i > 0:
        aligned_seq1 = ([True] * i) + aligned_seq1
        aligned_seq2 = ([False] * i) + aligned_seq2
       
    if j > 0: 
        aligned_seq1 = ([False] * j) + aligned_seq1
        aligned_seq2 = ([True] * j) + aligned_seq2
        
    if (ri < L1):
        aligned_seq1.extend([True] * (L1 - ri))
        aligned_seq2.extend([False]*(L1-ri))
        
    if rj < L2:
        aligned_seq2.extend([True] * (L2 - rj))
        aligned_seq1.extend([False]*(L2-rj))
        
      
    return aligned_seq1, aligned_seq2


def SW_alignment(sim_matrix, gap_penalty, only_score=False):
    
    '''
    Smith-Waterman pairwise alignment algorithm
    '''
    L1, L2 = sim_matrix.shape  
    H, T = SW_compute_score(sim_matrix, gap_penalty)
    
    if only_score == True: 
        return np.max(H)
    
    return SW_traceback(L1, L2, H, T)


def align2seq(align, seq):
    aligned_seq = []
    i = 0
    
    for j in range(len(align)):
        if align[j] == True:
            aligned_seq.append(seq[i])
            i += 1
        else: 
            aligned_seq.append('-')
    
    return ''.join(aligned_seq)


def local_pairwise_alignment(seq1, seq2, f1, f2, gap_penalty):
    '''
    Run local pairwise alignment for two sequences
    '''
    d = utils.pairwise_euclidean(f1, f2)
    d = np.exp(-d)
    d = stats.zscore(d)
    d = utils.context_based_similarity(d)
    
    align1, align2 = SW_alignment(d, gap_penalty)
    seq1 = align2seq(align1, seq1)
    seq2 = align2seq(align2, seq2)
    return seq1, seq2


def NW_compute_score(sim_matrix, gap_penalty):
    '''
    Compute Needleman-Wunsch score using dynamic programming algorithm
    '''
    L1, L2 = sim_matrix.shape
    H = np.zeros((L1+1, L2+1))
    H[0,:] = np.array([-i*gap_penalty for i in range(L2+1)])
    H[:,0] = np.array([-i*gap_penalty for i in range(L1+1)])
        
    T = np.zeros((L1+1, L2+1))
    T[0, 1:] = 3 #left 
    T[1:, 0] = 2#up
    
    for i in range(1, L1+1):
        for j in range(1, L2+1):
            options = []
            
            a = H[i-1, j-1] + sim_matrix[i-1, j-1]
            
            
            options.append((a, 4)) #diag
            
            b = H[i, j-1] - gap_penalty
            options.append((b, 3)) #left
             
            c = H[i-1, j] - gap_penalty
            options.append((c, 2)) #up
                         
            d = max(options)
            
            H[i, j] = d[0] 
            T[i, j] = d[1] 
                      
    return H, T

def NW_traceback(H, T):
    '''
    Traceback for Needleman-Wunsch pairwise alignment algorithm
    '''
    i, j = H.shape
    i = i-1 
    j = j-1
    aligned_seq1 = []
    aligned_seq2 = []
    
    while (i > 0 or j > 0):
        
        if T[i, j] == 4: #diag
            aligned_seq1.insert(0, True)
            aligned_seq2.insert(0, True)
            i -= 1 
            j -= 1 
            
        elif T[i, j] == 3: #left 
            aligned_seq1.insert(0, False)
            aligned_seq2.insert(0, True)
            j -= 1 
        elif T[i, j] == 2: #up 
            aligned_seq1.insert(0, True)
            aligned_seq2.insert(0, False)
            i -= 1
           
    return aligned_seq1, aligned_seq2


def NW_alignment(sim_matrix, gap_penalty, only_score=False):
    
    '''
    Needleman-Wunsch pairwise alignment algorithm
    '''
    H, T = NW_compute_score(sim_matrix, gap_penalty)
    
    if only_score == True: 
        return H[-1,-1]
    
    return NW_traceback(H, T)

#Cette fonction est column score, c'est un score binaire, le score est calculé pour chaque colonne similaire entre deux sequence (une referente, et l'autre celle qu'on veut tester).
def compare_two_alignments(tcoffee_path, input_aln, ref_aln, mode="column"):
    '''
    Comparing alterative alignments using t-coffee (to evaluate MSA alignment)
    '''
    cmd = [
            tcoffee_path,
            "-other_pg",
            "aln_compare",
            "-al1", ref_aln,
            "-al2", input_aln,
            "-compare_mode", mode,
        ]
        #print(file_name)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    retcode = process.wait()
    if retcode:
        error_msg = "Evaluation failed\nstderr:\n%s\n" % (stderr.decode("utf-8"))
        print(error_msg)
        return -1.0
    result_msg = "%s" % stdout.decode("utf-8")
    result_msg = result_msg.splitlines()[-1] #last line is the result
    score = float(result_msg.split()[3])
    return score
    
