import numpy as np
import pickle
import copy
import multiprocessing as mp
import os
from os.path import isfile, join
from math import log2, exp
from collections import Counter

def compute_distortion(c, z, m_c, mhat_z):
    # assert m_c[c].keys()==mhat_z[z].keys()
    distortion = 0
    for key, value in m_c[c].items():
        if mhat_z[z][key]==0:
            distortion = 2**13 # Inf
            return distortion
        else:
            div = m_c[c][key]/mhat_z[z][key]
            distortion += value*log2(div)
    return distortion


def compute_F(beta, pZ, pZ_condC, mhat_z, pC, m_c, Z, C):
    complexity = 0
    for z in Z:
        for c in C:
            if pZ_condC[c][z]!=0:
                div = pZ_condC[c][z]/pZ[z]
                complexity += pC[c]*pZ_condC[c][z]*log2(div)

    avg_distortion = 0
    for z in Z:
        for c in C:
            distortion = compute_distortion(c, z, m_c, mhat_z)
            avg_distortion += pC[c]*pZ_condC[c][z]*distortion

    return (complexity, avg_distortion, complexity+beta*avg_distortion)

def entropy_old(var):
    H = 0
    uniques = np.unique(var)
    for x in uniques:
        p = sum([avar==x for avar in var])/len(var)
        if p != 0:
            H += -p*log2(p)
    return H

def entropy(var):
    p = np.array([*Counter(var).values()])/len(var)
    H = sum(-p*np.log2(p))
    return H


 def compute_dic_multi(alpha, _dict):
    new_dict = {}
    for key, value in _dict.items():
        new_dict[key] = alpha*value
    return new_dict

def compute_dict_sum(_dict1, _dict2):
    assert _dict1.keys() == _dict2.keys()

    new_dict = {}
    for key in _dict1.keys():
        new_dict[key] = _dict1[key] + _dict2[key]

    return new_dict

def compute_JS(pi1, pi2, prob1, prob2):
    sum_prob = compute_dict_sum(compute_dic_multi(pi1,prob1), compute_dic_multi(pi2,prob2))
    return entropy([*sum_prob.values()])-pi1*entropy([*prob1.values()])-pi2*entropy([*prob2.values()])

def compute_d(beta, zi, zj, pzi, pzj, mhat_z, pc_condZ):
    tmp1 = compute_JS(pzi/(pzj+pzi), pzj/(pzj+pzi), mhat_z[zi], mhat_z[zj])
    tmp2 = compute_JS(pzi/(pzj+pzi), pzj/(pzj+pzi), pc_condZ[zi], pc_condZ[zj])
    return tmp1 - (1/beta)*tmp2

def iIB(C, Z, pC, beta, pZ_old, mhat_z_old, m_c):
    # Compute p(w|c) based on pZ and mhat_z
    pZ_condC = np.zeros((N, partitions)) # just to give the
    distortions = {}
    for c in C:
        distortions[c] = {}
        for z in Z:
            distortions[c][z] = compute_distortion(c, z, m_c, mhat_z_old)

    for z in Z:
        for c in C:
            pZ_condC[c][z] = pZ_old[z]* exp(-beta*distortions[c][z])

            normalization = 0
            for ztmp in Z:
                normalization += pZ_old[ztmp]* exp(-beta*distortions[c][ztmp])

            if normalization == 0:
                boolean = False
                return pZ_condC, pZ_old, mhat_z_old, False # we will not consider the first 3 values here
            else:
                pZ_condC[c][z] = pZ_condC[c][z]/normalization

    # compte pZ based on pZ_condC
    pZ_new = copy.deepcopy(pZ_old)
    for z in Z:
        pZ_new[z] = 0
        for c in C:
            pZ_new[z] += pZ_condC[c][z]*pC[c]

    # compute mhat_z based on pZ_condC and pZ
    mhat_z_new = {}
    for z in Z:
        mhat_z_new[z] = {}
        for key in m_c[0].keys():
            mhat_z_new[z][key] = 0
            for c in C:
                mhat_z_new[z][key] += m_c[c][key]*((pZ_condC[c][z]*pC[c])/pZ_new[z])

    return pZ_condC, pZ_new, mhat_z_new, True

def iIB_stopping_criteria(pZ_condC_old, pZ_condC_new, epsilon):
    stop = True
    for c in C:
        new, old = {}, {}
        for i, (value_new, value_old) in enumerate(zip(pZ_condC_new[c], pZ_condC_old[c])):
            old[i] = value_old
            new[i] = value_new
        JS = compute_JS(0.5, 0.5, new, old)
        if JS>epsilon:
            stop=False
    return not(stop)

# define mcu
def del_elt(old_list, i,j):
    _list = list(copy.deepcopy(old_list))
    if i<j:
        del _list[j]
        del _list[i]
    else:
        del _list[i]
        del _list[j]
    return _list

N = 330
distance_matrix = pickle.load(open('/private/home/rchaabouni/EGG_public/egg/zoo/color_signaling/data/distance_matrix', 'rb'))
# define chips
C = range(N)
Y = range(N)
pC = [1/float(N)]*N
m_c = pickle.load(open( "/private/home/rchaabouni/EGG_public/egg/zoo/color_signaling/data/IB_optim/m_c.p", "rb" ))

starting = 9
epsilon = 0.001
inverted_betas = list(np.arange(1.001,2,0.001)) + list(np.arange(2,11,1)) + [20, 40, 60, 80, 100] #+ [150,2**13]
betas = inverted_betas[::-1]

# Initialization
if starting == 0:
    print('here we are')
    ## Initialization if starting from the beginning of the model
    partitions = copy.deepcopy(N)
    Z = range(partitions)

    pZ = []
    mhat_z = {}
    pZ_condC = np.eye(N) #pZ_condC[ci][zj] = p(zj|ci)

    for i in range(N):
        pZ.append(copy.deepcopy(pC[i])) # p(zi)=p(ci)
        mhat_z[Z[i]] = copy.deepcopy(m_c[C[i]]) # mhat_z(u) = m_c(u) for each u

elif starting>0:
    ## Initialization if starting not from the beginning of the algorithm
    path_toprevious_sol = join('/private/home/rchaabouni/EGG_public/egg/zoo/color_signaling/data/IB_optim/solutions',\
                              f'betasol_{betas[starting-1]}.p')
    print('previous solution is saved in ', path_toprevious_sol)
    solution = pickle.load(open(path_toprevious_sol, 'rb'))

    pZ_condC = solution['pZ_condC']
    partitions = len(pZ_condC[1])
    Z = range(partitions)

    pZ = [0]*len(Z)
    for z in Z:
        for c in C:
            pZ[z] += pZ_condC[c][z]*pC[c]

    mhat_z = {}
    for z in Z:
        mhat_z[z] = {}
        for key in m_c[0].keys():
            mhat_z[z][key] = 0
            for c in C:
                mhat_z[z][key] += m_c[c][key]*((pZ_condC[c][z]*pC[c])/pZ[z])

# LOOP
for beta in betas[starting:]:
    beta = round(beta,4)
    print(f'beta={beta}')
    # compute another DeltaL from the beginning
    ## Cheating for the first value because we have it
    tmp_path = f"/private/home/rchaabouni/EGG_public/egg/zoo/color_signaling/data/IB_optim/DeltaL_{beta}.p"
    if os.path.exists(tmp_path):
        # load it
        print('loading delta')
        DeltaL = pickle.load(open(tmp_path, "rb" ))
    else:
        # compute it
        print('computing delta')
        pc_condZ = {}
        for z in Z:
            pc_condZ[z] = {}
            for c in C:
                pc_condZ[z][c] = (pZ_condC[c][z]*pC[c])/pZ[z]

        def fill_matrix(i, Z=Z, pZ=pZ, beta=beta, mhat_z=mhat_z, pc_condZ=pc_condZ):
            new_matrix = np.zeros((len(Z), len(Z)))
            zi = Z[i]
            for j, zj in enumerate(Z[:i]):
                new_matrix[zi,zj] = (pZ[zi]+ pZ[zj])*compute_d(beta, zi, zj, pZ[zi], pZ[zj], mhat_z, pc_condZ)
            return new_matrix

        count = len(Z)
        try:
            pool = mp.Pool(count)
            results = pool.map_async(fill_matrix, [i for i in  range(len(Z)-1,-1,-1)]).get()
        finally:
            pool.close()
            pool.join()

        DeltaL = sum(results)
        print('important test', len(DeltaL))
        pickle.dump(DeltaL, open(tmp_path, "wb" ))

    while len(Z)>1:
        print(f'aIB algo with {len(Z)} clusters')
        # find clusters
        ## DeltaL is a lower matrix
        m = DeltaL.shape[0]
        r,c = np.tril_indices(m,-1) # get lower matrix
        idx = DeltaL[r,c].argpartition(1)[0]
        i, j = r[idx], c[idx]

        # Merge
        partitions -= 1
        ## New Z
        Z_init = range(partitions)
        ## New pZ_condC
        pZ_condC_init = np.zeros((N, partitions))
        for c in C:
            tmp_val = pZ_condC[c][Z[i]] + pZ_condC[c][Z[j]]
            pZ_condC_init[c][:-1] = del_elt(pZ_condC[c],i,j)
            pZ_condC_init[c][-1] = tmp_val
        ## New pZ
        tmp_val1 = copy.deepcopy(pZ[i]+pZ[j])
        pZ_init = del_elt(pZ,i,j)
        pZ_init = pZ_init + [tmp_val1]
        ## New mhat_z
        Pi = pZ[i]/tmp_val1
        Pj = pZ[j]/tmp_val1
        tmp = compute_dict_sum(compute_dic_multi(Pi,mhat_z[i]), compute_dic_multi(Pj,mhat_z[j]))
        ### Delete keys i and j
        mhat_z_tmp = copy.deepcopy(mhat_z)
        del mhat_z_tmp[i]
        del mhat_z_tmp[j]
        # Change the name of keys
        mhat_z_init = {}
        count = 0
        for key, value in mhat_z_tmp.items():
            mhat_z_init[count] = value
            count += 1
        mhat_z_init[partitions-1] = tmp

        # iIB
        nonstop = True
        while nonstop:
            print('iIB algo')
            pZ_condC_new, pZ_new, mhat_z_new, boolean = iIB(C, Z_init, pC, beta, pZ_init, mhat_z_init, m_c)
            if not(boolean):
                # control if degenerated solution arises because of high beta value
                print('boolean is false')
                break

            print('computing iIB stopping criteria')
            nonstop = iIB_stopping_criteria(pZ_condC_init, pZ_condC_new, epsilon)
            if not(nonstop):
                break
            pZ_condC_init = pZ_condC_new
            pZ_init = pZ_new
            mhat_z_init = mhat_z_new

        # Update?
        comp_old, dist_old, F_old = compute_F(beta, pZ, pZ_condC, mhat_z, pC, m_c, Z, C)
        comp_new, dist_new, F_new = compute_F(beta, pZ_init, pZ_condC_init, mhat_z_init, pC, m_c, Z_init, C)
        print(comp_new, dist_new, F_new)

        if F_new < F_old:
            print('the new solution is better. We are updating')
            # The new one is better so use it at init for the next step
            pZ_condC = pZ_condC_init
            pZ = pZ_init
            mhat_z = mhat_z_init
            Z = Z_init
            # Update DeltaL
            DeltaL_new = copy.deepcopy(DeltaL)
            to_remove = [i,j]
            to_remove.sort()
            DeltaL_new = np.delete(DeltaL_new, to_remove, 1)
            DeltaL_new = np.delete(DeltaL_new, to_remove, 0)
            # Add new row corresponding to the new cluster (only a row as the matrix is symmetric)
            pc_condZ = {}
            for z in Z:
                pc_condZ[z] = {}
                for c in C:
                    if pZ[z]==0:
                        if pZ_condC[c][z]==0:
                            pc_condZ[z][c] = 0
                        else:
                            print('whaaaaat???')
                    else:
                        pc_condZ[z][c] = (pZ_condC[c][z]*pC[c])/pZ[z]
            zk = Z[-1]
            array_toadd = []
            for count, zl in enumerate(Z[:partitions-1]):
                array_toadd.append((pZ[zk]+ pZ[zl])*compute_d(beta, zk, zl, pZ[zk], pZ[zl], mhat_z, pc_condZ))

            array_toadd += [0.0]
            DeltaL_new = np.hstack( (DeltaL_new, np.array([0]*len(DeltaL_new)).reshape(len(DeltaL_new), 1) ) )
            DeltaL_new = np.vstack( (DeltaL_new, np.array(array_toadd).reshape(1,len(array_toadd))) )

            DeltaL = DeltaL_new
            # Go back to merging clusters
            print('merging more')

        else:
            partitions = len(Z)
            print(f'we find a solution for beta={beta}. We save it and do the process again after annealing beta')
            # save everything corresponding to that beta
            dict_tosave = {'complexity': comp_old, 'distortion': dist_old, 'F': F_old, 'pZ_condC': pZ_condC}
            pickle.dump(dict_tosave, open(f"/private/home/rchaabouni/EGG_public/egg/zoo/color_signaling/data/IB_optim/solutions/betasol_{beta}.p", "wb" ))
            break
