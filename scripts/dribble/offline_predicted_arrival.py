"""Offline check of the predicted-arrival ball reward on saved probe rollouts (no sim).
Usage: python scripts/dribble/offline_predicted_arrival.py <dir with probe_ref1*.npz>"""
import numpy as np,sys
S=sys.argv[1]; dt=0.02; SIG=0.20
m=np.load('motions/master-aligned/master_001675_001884_unitree_g1.npz',allow_pickle=True)
refp=m['body_pos_w'][:,0,:2]; refb=m['ball_pos_w'][:,:2]
c=m['dribble_cg_contact']>0; ev=np.flatnonzero(c&~np.r_[False,c[:-1]])
def rot(th): c_,s_=np.cos(th),np.sin(th); return np.array([[c_,-s_],[s_,c_]])
rows=[]
for p in ('probe_ref1.npz','probe_ref1_n0.2.npz','probe_ref1_n0.4.npz'):
    z=np.load(f'{S}/{p}',allow_pickle=True)
    rxy=z['ts_root_xy'];bxy=z['ts_ball_xy'];al=z['ts_alive'];fr=z['ts_frame'];n=rxy.shape[1]
    for i in range(n):
        T=int(al[:,i].sum())
        if T<100: continue
        r=rxy[:T,i]; b=bxy[:T,i]
        a=r[60]-r[5]; bb=refp[60]-refp[5]; th=np.arctan2(a[1],a[0])-np.arctan2(bb[1],bb[0]); R=rot(th)
        node=lambda k:(refb[ev[k]]-refp[0])@R.T+r[0]          # episode-fixed world node (not live-root)
        vel=np.vstack([np.zeros((1,2)),np.diff(b,axis=0)/dt])
        for k in (1,2):                                        # transition into event k (node k); ev1@47, ev2@67
            tk=ev[k]
            if tk>=T-1: continue
            nk=node(k)
            for lead in (10,20):                               # reward evaluated 0.2 s / 0.4 s before node time
                t=tk-lead
                if t<5: continue
                pred=b[t]+vel[t]*lead*dt                       # constant-velocity roll (sim decays ~3%/s)
                e_pred=np.linalg.norm(pred-nk); e_act=np.linalg.norm(b[tk]-nk); e_still=np.linalg.norm(b[t]-nk)
                rows.append((k,lead,np.linalg.norm(vel[t]),e_pred,e_act,e_still))
R_=np.array(rows)
print('samples',len(R_))
for k in (1,2):
  for lead in (10,20):
    s=R_[(R_[:,0]==k)&(R_[:,1]==lead)]
    mv=s[:,2]>0.3
    print(f'node ev{k}, evaluated {lead*dt:.1f}s early: n={len(s)}, ball moving {mv.sum()}')
    if mv.sum()>3:
        d=np.abs(s[mv,3]-s[mv,4]); print(f'    moving: |pred_err - actual_err| median {np.median(d):.3f} m (pred_err median {np.median(s[mv,3]):.2f}, actual median {np.median(s[mv,4]):.2f}); corr(pred_err,actual_err)={np.corrcoef(s[mv,3],s[mv,4])[0,1]:.2f}')
        rew=np.exp(-s[mv,3]**2/SIG**2); print(f'    reward(moving) p10/med/p90 {np.percentile(rew,[10,50,90]).round(3).tolist()}')
    st=~mv
    if st.sum(): rew=np.exp(-s[st,3]**2/SIG**2); print(f'    still ball: n={st.sum()} dist-to-node median {np.median(s[st,3]):.2f} m, reward p10/med/p90 {np.percentile(rew,[10,50,90]).round(3).tolist()}')
