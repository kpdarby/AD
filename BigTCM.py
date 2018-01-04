#if __name__ == '__main__':

import BigT
from ddm_like import wfpt

#def luce(x, p=1.0):
#    xp = x**p
#    return xp/xp.sum()

class BigTCM(object):
    def __init__(self, items, vlen=150):

        # allocate big T and M
        self.bt = BigT.BigT(vlen)
        self.M = items[0].T*self.bt.T

    # function for presenting item
    def present_item(self, item, alphaT = 1., alphaM=1., dur=2.5):
        # bind that item to context
        if alphaM > 0.0:
            self.M += alphaM*item.T*self.bt.T

        # update context
        # present an item for a timestep
        self.bt.update_t(item=item, alpha = alphaT)

        # drift for some time
        self.bt.update_t(dur=dur, alpha = alphaT)



    def calc_cont_recog_like(self, list_def, responses, rts=None, save_posts=False):
        # loop over items
        if rts is None:
            rts = [None]*len(responses)
        log_like = []
        posts = []
        for i,r,rt in zip(list_def,responses,rts):


            # present the item
            self._present_item(i, self.params['rho'], alpha=self.params['alpha'])

            # get the strength
            strength = self.mem_strength

            # get strength and calc like
            if r > 0 and rt<5.0:
                # only calc if they made a response
                ll,post = self._recog_like_ddm(r, strength, rt, save_posts)
                log_like.append(ll)
                posts.append(post)

        return log_like,posts


    def _recog_like_ddm(self, resp, strength, rt, save_posts=False):

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

        # get full pdist if save posts.
        if save_posts:
            posts = [wfpt(t,v,self.params['a'],w) for t in self.tposts]
            if self.params['d'] < 1.0:
                posts = [posts[i]*self.params['d'] +
                         wfpt(t,0.0,self.params['a_d'],w)*(1-self.params['d'])
                         for i,t in enumerate(self.tposts)]
        else:
            posts = None

        # return the log like
        return np.log(like), posts