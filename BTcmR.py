import numpy as np
from ddm_like import wfpt
import BigT

#import matplotlib

class BigTCM(object):

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

    def __init__(self, listlen=None, nitems=None, params=None):

        # set up the tposts for ppd
        self.tposts = np.linspace(0, 5, 200)

        # save the nitems
        self.listlen = listlen
        self.nitems = nitems

        self.items = np.asmatrix(np.eye(nitems))

        # process the params
        # start with defaults
        p = dict(**self.default_params)

        if params is not None:
            # get provided vals
            p.update(params)
        self.params = p

        # allocate big T
        self.bt = BigT.BigT(nitems, T_full_ind=slice(None, None, 6), s_toskip=12, ntau=90)

        # present some items that fill big T but don't associate M
        dist_ind = self.items.shape[0]
        for i in range(nitems - listlen):
            dist_ind -= 1

            self._present_item(item = self.items[dist_ind], alphaT = self.params['alphaT'], dur=5.)


    # function for presenting item
    def _present_item(self, item, beta=.5, alphaT = 1., dur=2.5, is_old=False):

        # replicate item to fill matrix matching T
        self.repItem = np.tile(item, self.bt.T_full.shape[0])
        items_sigged = np.asmatrix(self.repItem.A.flatten())

        # find similarity between replicated item and current T
        this_strength = items_sigged*self.bt.T.T
        # take the log of strength
        #this_strength_logged = np.log((this_strength*100.)+1.)
        # scale strength
        #this_strength_scaled = this_strength_logged*self.params['scale']
        this_strength_scaled = this_strength*self.params['scale']
        # clip strength
        #self.mem_strength = np.clip(this_strength_scaled, 0, self.params['clipper'])
        self.mem_strength = this_strength_scaled
        if is_old:
            self.mem_strength += self.params['alpha']

        # update context
        # present an item for a timestep
        self.bt.update_t(item=item, alpha = alphaT)

        # drift for some time
        self.bt.update_t(dur=dur, alpha = alphaT)
  

    def calc_cont_recog_like(self, list_def, responses, durs, rts=None):
        # loop over items
        if rts is None:
            rts = [None]*len(responses)
        log_like = []

        presented = []
        for i,r,rt,dur in zip(list_def,responses,rts,durs):

            # present tiem to T
            is_old = i in presented
            self._present_item(item = self.items[i-1], alphaT = self.params['alphaT'], dur=dur, is_old=is_old)
            presented.append(i)

            # get the strength
            strength = self.mem_strength

            # get strength and calc like
            if r > 0 and rt<5.0:
                # only calc if they made a response
                ll = self._recog_like_ddm(r, strength, rt)
                log_like.append(ll)

        return log_like

    def _recog_like_ddm(self, resp, strength, rt):

        # diffusion is to lower boundary, make that old
        v = self.params['nu'] - strength
        w = self.params['w']
        if resp != 1:
            # new response
            # flip the drift
            v = -v
            # flip the start point
            w = 1-w

        w_k = self.params['w_k']
        if not w_k is None:
            w_k = np.exp(w_k)

        # calc the likelihood
        like = wfpt(rt-self.params['t0'],v,self.params['a'],
                    w_mode=w, w_k=w_k)

        if self.params['d'] < 1.0:
            # calc like with no evidence
            guess_like = wfpt(rt-self.params['t0'],0.0,self.params['a_d'],
                              w_mode=w, w_k=w_k)
            like = like*self.params['d'] + guess_like*(1-self.params['d'])


        # return the log like
        return np.log(like)
        
