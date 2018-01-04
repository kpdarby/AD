import numpy as np
#from joblib import Parallel,delayed
from scoop import futures
import sys
import os
import pandas as pd
from glob import glob

from RunDEMC import Model, Param, dists
from RunDEMC import Hierarchy, HyperPrior
from RunDEMC import save_results

import BigT
import BTcmR as BigTCM



data_dir = '../data'
data_path = os.path.join(data_dir,'IMAS_SKlogs')

# set up the params
default_params = {
        # BigTCM
        'alphaT': 1.0,
        'alpha': 1.0,
        'scale': 1.0,

        # ddm params
        'a': 2.0,
        'w': .25,
        'w_k': None,
        'd': 1.0,
        'a_d': None,
        'nu': 0.0,
        't0': .25,
}


# set up the data
def load_subj_data_file(subj_file):
    x = pd.read_csv(subj_file,skiprows=3,sep='\\t',skipfooter=13,engine='python')
    # turn into responses and unique stim
    ind = {}
    cind = 1
    dat = {'listdef':[]}
    # find time of last trial (to append to time list for final real trial duration)
    last_rest_time = x.Time.values[-1]
    
    for w in x.Event[x.Event != 'sndREST']:
        if not ind.has_key(w):
            ind[w] = cind
            cind += 1
        dat['listdef'].append(ind[w])
    dat['resps'] = list(x.Response[x.Event != 'sndREST'])
    dat['rts'] = list(x.RXNTime[x.Event != 'sndREST'])
    dat['time'] = list(x.Time[x.Event != 'sndREST']) + [last_rest_time]
    dat['timediff'] = list(np.array(np.diff(dat['time'])))
    dat['type'] = list(x.Type[x.Event!='sndREST'])
    return dat

def load_subj_data(subj):
    subj_files = glob(os.path.join(data_path,'%s*.xls'%subj))
    subj_files.sort()
    dat = [load_subj_data_file(sf) for sf in subj_files]
    return dat


def eval_mod(params, param_names, bdat=None, verbose=False):
    # use global dat if none based in
    if bdat is None:
        bdat = dat

    # do logit transforms as necessary
    
    # turn param list into dict
    mod_params = default_params.copy()
    mod_params.update({x: params[n] 
                       for n,x in enumerate(param_names)})


    ll = 0.0
    for d in bdat:
        if verbose:
            sys.stdout.write('.')
            sys.stdout.flush()
        # define the model class and calc likelihoods based on data_info

        btcm = BigTCM.BigTCM(nitems = max(d['listdef'])*2, listlen = max(d['listdef']), params = mod_params)

        lls = btcm.calc_cont_recog_like(list_def = d['listdef'], responses = d['resps'],
                                        rts = d['rts'], durs = d['timediff'])
        ll += np.sum(lls)

    return ll


def eval_fun(pop, *args):
    # call each particle in parallel
    bdat = args[1]
    pnames = args[2]
    likes = list(futures.map(eval_mod, [indiv for indiv in pop],
                           [pnames]*len(pop), [bdat]*len(pop)))

    return np.array(likes)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str,
                        help="subject identifier")
    parser.add_argument("run_name", type=str,
                        help="run identifier")
    args = parser.parse_args()

    # set info
    subj = [int(s) for s in (args.subject).split(',')]
    subj_name = '_'.join([str(s) for s in subj])
    run_name = args.run_name

    print('Subject:',subj)
    print('Run:',run_name)

    # proc the data
    #initialize data structures
    out_path = 'sims/%s' % run_name
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path,'%s_%s.tgz'%(run_name,subj_name))

    dat = {}
    for s in subj:
        if s < 10:
            # it's a diagnosis
            odat = pd.read_csv('proc_diag.csv')
            subjs = list(odat.subject[odat.BL_DXGrp_2==s])
            print(subjs)

            # load each one
            for ss in subjs:
                dat[ss] = load_subj_data(ss)
        else:
            # load the data
            if dat.has_key(s):
                # we're running cross validation
                cross_validate = True
                # remove that subj
                dat.pop(s,None)
                print("Removed %d for cross validation."%s)
            else:
                # just add it like normal
                dat[s] = load_subj_data(s)

    # get the min_RT
    min_RT = np.min([np.min(np.array(d['rts'])[np.array(d['rts'])>0.0])
                     for sub in dat for d in dat[sub] ])
  

    for s in dat.keys():
                                # Append a new model, note the use of the hyperpriors
                                params = [

                                        # sig_b is the sigmoid transition point
                                        #Param(name='sig_b',
                                        #                prior=dists.uniform(0., 15.0), # change to number of tau stars
                                        #                ),

                                        # new item strength
                                        Param(name='alpha',
                                                display_name=r'$\alpha$',
                                                prior=dists.uniform(0,10.0),
                                                #init_prior=dists.trunc_normal(mean=.25,std=.5,lower=0,upper=5)
                                                ),
                                        Param(name='nu',
                                                display_name=r'$\nu$',
                                                prior=dists.uniform(0,10.0),
                                                #init_prior=dists.trunc_normal(mean=.25,std=.5,lower=0,upper=5)
                                                ),
                                        # threshold
                                        Param(name='a',
                                                display_name=r'$a$',
                                                prior=dists.uniform(0,10.0),
                                                #init_prior=dists.trunc_normal(mean=.25,std=.5,lower=0,upper=5)
                                                ),
                                        # starting point
                                        Param(name='w',
                                                display_name=r'$w$',
                                                prior=dists.uniform(0,1.0),
                                                #init_prior=dists.trunc_normal(mean=.25,std=.5,lower=0,upper=5)
                                                #transform=dists.invlogit
                                                ),
                                        # non-decision time
                                        Param(name='t0',
                                                display_name=r'$t_0$',
                                                prior=dists.uniform(0,min_RT),
                                                #init_prior=dists.trunc_normal(mean=.1,std=.25,lower=0,upper=min_RT)
                                                ),
                                        # strength scale
                                        Param(name='scale',
                                                display_name=r'$scale$',
                                                prior = dists.uniform(0,10)),
                                          ]
                        
                                #grab the param names
                                pnames = [p.name for p in params]
                                m = Model(s,params=params,
                                                 like_fun=eval_fun,
                                                 like_args=(s,dat[s],pnames),
                                                 verbose=True)
                        
                                m(50, burnin = True)
                                save_results(out_file, m)
                                m(25, burnin = True)
                                save_results(out_file, m)
                                m(25, burnin = True)
                                save_results(out_file, m)
                                m(25, burnin = True)
                                save_results(out_file, m)
                                m(25, burnin = True)
                                save_results(out_file, m)
                                m(25, burnin = True)
                                save_results(out_file, m)
                                m(25, burnin = True)
                                save_results(out_file, m)
                    
            
