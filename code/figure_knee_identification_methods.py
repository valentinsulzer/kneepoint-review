
from pathlib import Path

# standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GPs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import config
# colours
yellow = np.array([1, .67, .14])
lgrey = np.array([.7, .7, .7])
lblue = np.array([.557, .729, .898])
jade = np.array([0, .66, .436]) # statue green
blue = np.array([.057, .156, .520]) # hey there mr blue
brown = np.array([.515, .158, .033]) # did someone order CDM?
red = np.array([.85, .20, 0]) # tikka masala
gold = np.array([1, .67, .14]) # Staffordshire hoard
claret = np.array([.429, .073, .238]) # claret
grey = np.array([.585, .612, .675]) # library grey
grey2 = np.array([233, 235, 238]) / 255 # library grey with 0.2 transparency
black = np.array([0,0,0]) # this is a black

# maths functions
from numpy import matmul as mm  # matrix multiplication
from numpy.linalg import inv    # matrix inversion
from scipy.interpolate import UnivariateSpline as spline

def ols(x,y):
    """
    Ordinary least squares fitting
    """
    # add the column of ones as the bias
    X = np.concatenate( ( np.ones((x.shape[0],1)), x ),
                        axis=1)
    # full calculation
    w = mm( inv( mm(X.T,X) ), mm(X.T,y) )
    
    # return w
    return w

def knee_point_identification(ax,time,capacity,colour):
    """
    Function identifies the position of the knee point on the capacity
    time curve. This function will take several assumptions about the 
    general shape, but should be applicable to any curves with a flat
    profile in early life followed by a dramatic loss of health.
    
    The process finds the meeting point of two linear extrapolations 
    from early and late life, then uses an angle bisector to find the 
    knee point on the actual curve.
    """
    ax.plot(time,capacity,color=colour)
    ax.set_ylim(75,100)
    
    # guarantee the inputs are column vectors
    t_scale = time.max()
    time = time.reshape((time.shape[0],1))/time.max()
    capacity = capacity.reshape((capacity.shape[0],1))
    
    # ============== EARLY LIFE LINEAR MODEL ================= #
    # find early life data (assumed first half of points)
    n = np.floor(time.shape[0]/2).astype(int)
    t_early = time[:n,:].reshape((n,1))
    q_early = capacity[:n,:].reshape((n,1))
    
    # early life model
    w_early = ols(t_early,q_early)
    q_early = mm(np.hstack((np.ones(time.shape),time)),w_early)
    
    ax.plot(time*t_scale,q_early,'--',color=grey)
    
    # ============== LATE LIFE LINEAR MODEL ================= #
    # find late life data (assumed last 5 points)
    n = time.shape[0]
    t_late = time[n-50:,:].reshape((50,1))
    q_late = capacity[n-50:,:].reshape((50,1))
    
    # late life model
    w_late = ols(t_late,q_late)
    q_late = mm(np.hstack((np.ones(time.shape),time)),w_late)
    
    ax.plot(time*t_scale,q_late,'--',color=grey)
    
    # ================== ANGLE BISECTOR ===================== #
    # intersection point
    t_int = (w_late[0]-w_early[0])/(w_early[1]-w_late[1])
    q_int = w_early[0] + w_early[1]*t_int
    
    # bisector gradient
    m_b = np.tan( (np.arctan(-1/w_early[1]) + np.arctan(-1/w_late[1]) ) /2 )

    # smoothed spline (needed later)
    spl_cap = spline(time,capacity,k=5)
    t_spl = np.linspace(time.min(),time.max(),1000).reshape((1000,1))
    q_spl = spl_cap(t_spl)
    q_bisector = (m_b*t_spl*t_scale) + q_int - (m_b*t_int*t_scale)
    
    ax.plot(t_spl[650:850]*t_scale,q_bisector[650:850],'--',color=grey)
    
    # ================= FINAL KNEE POINT ==================== #    
    # make sure there is a result, otherwise return inf
    if t_spl[ (q_spl>q_bisector) ].shape[0]>0:
        # knee time
        t_knee = np.max( t_spl[ (q_spl>q_bisector) ] )
        # knee point
        q_knee = spl_cap(t_knee)
    
    else:
        t_knee = np.inf
        q_knee = np.inf
    
    ax.scatter(t_knee*t_scale,q_knee,color="black",marker='s')
    
    return t_knee*t_scale, q_knee

# BACON WATTS
from scipy.optimize import minimize
def bacon_watts_knee(ax,time,capacity,colour):
    ax.plot(time,capacity,color=colour)
    ax.set_ylim(75,100)
    
    # guarantee the inputs are column vectors
    time = time.reshape((time.shape[0],1))
    capacity = capacity.reshape((capacity.shape[0],1))
    
    def bw_func(x,p):
        r = x-p[3]
        y =   p[0] \
            + p[1]*r \
            + p[2]*r*np.tanh(r/(10**-5))
        return y
    def loss_func(p):
        L = (capacity-bw_func(time,p))**2
        return L.sum()
    p0 = np.array([100,-.01,-.01,400.0])
    M = minimize(loss_func,p0,method='Nelder-Mead')
    
    # plot the bacon watts method with the parameters
    ax.plot(time,bw_func(time,M.x),'--',color=grey)
    
    # find the knee point time
    t_knee = M.x[3]
    
    # find the capacity of the knee point
    spl_cap = spline(time,capacity,k=5)
    q_knee = spl_cap(t_knee)
    
    # plot knee point and line used for it
    ax.scatter(t_knee,q_knee,color="black",marker='s')
    ax.plot(np.array([t_knee,t_knee]),np.array([80,98]),
            '--',color=grey)
    
    return t_knee, q_knee

def d2qdt2(t,q):    
    dqdt = np.diff(q,axis=0) / np.diff(t,axis=0)
    d2q = np.diff(dqdt,axis=0) / np.diff(t[1:].reshape((t.shape[0]-1,1)),axis=0)
    return d2q
    
def kneedle_identification(ax,time,capacity,colour):
    ax.plot(time,capacity,color=colour)
    ax.set_ylim(75,100)
    
    # guarantee the inputs are column vectors
    time = time.reshape((time.shape[0],1))
    capacity = capacity.reshape((capacity.shape[0],1))

    # actual kneedle algorithm from Satopaa
    
    # nomalise the capacity curve
    d_time = time.max()-time.min();
    d_cap = capacity.max()-capacity.min();
    q_norm = (capacity - capacity.min())/d_cap;
    t_norm = (time - time.min())/d_time;
    
    # store 1/sqrt(2) = cos(\theta)
    sqrt2 = .5**.5;
    
    # create new x variable
    x = sqrt2*t_norm - sqrt2*q_norm;
    # rotate to make the bisector the x-axis from 0 to 1;
    y = sqrt2*t_norm + sqrt2*q_norm;
    
    # find maximum deviation
    indx = np.argmax(y)
    
    x_indx = x[indx]; y_indx = y[indx];
    
    t_norm_knee = sqrt2*(x_indx+y_indx);
    q_norm_knee = sqrt2*(-x_indx+y_indx);
    
    t_knee = (t_norm_knee*d_time) + time.min()
    q_knee = (q_norm_knee*d_cap) + capacity.min()
    
    x_axis_x = sqrt2*(x+y_indx);
    x_axis_y = sqrt2*(-x+y_indx);
    x_axis_t = (x_axis_x*d_time) + time.min()
    x_axis_q = (x_axis_y*d_cap) + capacity.min()
    
    # plot an example x-axis for the kneedle method
    ax.plot(x_axis_t[40:]-140,x_axis_q[40:]-5,'--',color=grey)
    
    # plot the 45 degree bisector (this is not automated because I want to 
    # make something that can be understood on Figure 4)
    ax.plot(np.array([262,365]),np.array([86.8,93.53]),'--',color=grey)
    
    # mark the knee point
    ax.scatter(t_knee,q_knee,color="black",marker='s')
    ax.plot(np.array([t_knee,t_knee]),np.array([80,98]),
            '--',color=grey)
    # ax.text(-10,85,'peak height from chord',color=grey)
    
    return t_knee, q_knee
    
def diao_knee(ax,time,capacity,colour):
    ax.plot(time,capacity,color=colour)
    ax.set_ylim(75,100)
    
    # guarantee the inputs are column vectors
    time = time.reshape((time.shape[0],1))
    capacity = capacity.reshape((capacity.shape[0],1))
    
    # ============== EARLY LIFE LINEAR MODEL ================= #
    # find early life data (assumed first half of points)
    n = np.floor(time.shape[0]/2).astype(int)
    t_early = time[:n,:].reshape((n,1))
    q_early = capacity[:n,:].reshape((n,1))
    
    # early life model
    w_early = ols(t_early,q_early)
    q_early = mm(np.hstack((np.ones(time.shape),time)),w_early)
    
    ax.plot(time,q_early,'--',color=grey)
    
    # ============== LATE LIFE LINEAR MODEL ================= #
    # find late life data (assumed last 5 points)
    n = time.shape[0]
    t_late = time[n-50:,:].reshape((50,1))
    q_late = capacity[n-50:,:].reshape((50,1))
    
    kernel = RBF(1e2, (8e1, 1e4))
    gp = GaussianProcessRegressor(kernel=kernel)
    gp.fit(time, capacity)
    t_gpm = np.linspace(time.min(),time.max(),1000).reshape((1000,1))
    q_gpm = gp.predict(t_gpm, return_std=False)
    
    d2q = d2qdt2(t_gpm,q_gpm)
    knee_indx = np.argmin(d2q[500:])
    t_late = t_gpm[knee_indx+495:knee_indx+505]
    q_late = q_gpm[knee_indx+495:knee_indx+505]
    
    
    w_late = ols(t_late,q_late)
    q_late = mm(np.hstack((np.ones(time.shape),time)),w_late)
    
    ax.plot(time,q_late,'--',color=grey)
    
    # ================== KNEE POINT CALC ===================== #
    # intersection point
    t_knee = (w_late[0]-w_early[0])/(w_early[1]-w_late[1])
    
    spl_cap = spline(time,capacity,k=5)
    q_knee = spl_cap(t_knee)
    
    ax.scatter(t_knee,q_knee,color="black",marker='s')
    ax.plot(np.array([t_knee,t_knee]),np.array([80,98]),
            '--',color=grey)
    
    return t_knee, q_knee

def zhang_knee(ax,time,capacity,dqdv,colour):
    ax_in = ax.inset_axes([50,82,310,11], transform=ax.transData)
    ax.plot(time,capacity,color=colour)
    ax.set_ylim(75,100)
    
    # guarantee the inputs are column vectors
    time = time.reshape((time.shape[0],1))
    y = dqdv.reshape((dqdv.shape[0],1))
    
    ax_in.plot(time,y,color=colour)
    
    # ============== EARLY LIFE LINEAR MODEL ================= #
    # find early life data (assumed first half of points)
    n = np.floor(time.shape[0]*2/3).astype(int)
    t_early = time[:n,:].reshape((n,1))
    y_early = y[:n,:].reshape((n,1))
    
    # early life model
    w = ols(t_early,y_early)
    
    t = np.linspace(time.min(),time.max(),1000)
    y_early_mod = w[0] + w[1]*t
    
    ax_in.plot(t,y_early_mod,'--',color=grey)
    
    y_low = y_early_mod - 0.2
    y_high = y_early_mod + 0.2
    
    ax_in.fill_between(t,y_low,y_high,color=grey2)#,alpha=.2)
    
    spl_y = spline(time,y,k=5)
    t_late = t[500:]
    y_late = spl_y(t_late)
    
    t_late = t_late[ y_late<y_low[500:] ]
    
    t_knee = t_late.min()
    y_knee = spl_y(t_knee)
    
    ax_in.scatter(t_knee,y_knee,color="black",marker='s')
    ax_in.plot(np.array([t_knee,t_knee]),np.array([3.1,6.3]),
            '--',color=grey)
    
    # plot controls
    ax_in.set_yticks([])
    ax_in.set_xticks([])
    ax_in.set_xlabel('cycle number')
    ax_in.set_ylabel('dq/dv (%/V)')
    
    spl_cap = spline(time,capacity,k=5)
    q_knee = spl_cap(t_knee)
    
    ax.scatter(t_knee,q_knee,color="black",marker='s')
    ax.plot(np.array([t_knee,t_knee]),np.array([85,95]),
            '--',color=grey)
    
    return t_knee,q_knee


def dq2dt2(ax,t,q,colour):    
    dqdt = np.diff(q,axis=0) / np.diff(t,axis=0)
    
    d2qdt2 = np.diff(dqdt,axis=0) / np.diff(t[1:],axis=0)
    ax.plot(t[4:],d2qdt2[2:]*1000,color=colour)

path = Path().cwd() / "data"
m = pd.read_csv(path / 'severson2019_b2c30_health_fig04.csv')

t = np.array(m['cycs']).reshape((m['cycs'].shape[0],1))
q = np.array(m['q']).reshape((m['cycs'].shape[0],1))
dqdv = np.array(m['dqdv']).reshape((m['dqdv'].shape[0],1))

σ = .0001
# qσ = q + (np.random.randn(q.shape[0],1)*σ)

# smooth capacities
kernel = RBF(1e2, (8e1, 1e4)); RBF()
gp = GaussianProcessRegressor(kernel=kernel)
gp.fit(t, q)
t_gpm = np.linspace(t.min()+50,t.max(),1000).reshape((1000,1))
q_gpm = gp.predict(t_gpm, return_std=False)

# KNEE DEFINITION ============================ FIGURE 3
fig, ax = plt.subplots(2,1,figsize=(config.FIG_WIDTH*1,config.FIG_HEIGHT*2))

# profile
# ax[0].plot(t,qσ,color=jade)
ax[0].plot(t,q,color=blue)
ax[0].set_ylabel('Capacity retention (%)')
ax[0].scatter(365,93.5,color="black")
ax[0].text(365,92.5,'Visual knee point',color="black",
          horizontalalignment='right',verticalalignment='top')
ax[0].scatter(435,88,color=gold)
ax[0].text(435,87,'Mathematical knee point',
           color=gold,
           horizontalalignment='right',verticalalignment='top')
ax[0].set_ylim([75, 100])

# calculating the knee points using second derivative

# first, this is the REAL (i.e. measured) capacity curve, but downselected
# to help the visibility of the noise issue:
n0=0; n1=508; n2=5;
dq2dt2(ax[1],t[np.array(range(n0,n1,n2))],q[np.array(range(n0,n1,n2))],grey)

# second, this is a smoothed capacity curve which produces a nice results:
dq2dt2(ax[1],t_gpm,q_gpm,blue)

# add text to label the two curves
ax[1].text(50,-1.8,'Observed capacity',color=grey)
ax[1].text(50,-2.2,'Smoothed capacity',color=blue)

ax[1].set_ylabel('Second derivative of retention/1000 (%)')

ax[1].plot([365, 365],[-2.5, .5],'--',color="black")
ax[1].plot([435, 435],[-2.5, .5],'--',color=gold)
# set limit to focus on the smoothed result
ax[1].set_ylim([-3, 1])

# # UNCOMMENT BELOW TO SEE NOISE IMPACT
# σ = .00007
# qσ_gpm = q_gpm + (np.random.randn(q_gpm.shape[0],1)*σ)
# dq2dt2(ax[1],t_gpm,qσ_gpm,jade)
# ax[1].text(10,-2.50,'Added noise, σ='+str(σ)+'% capacity',color=jade)


for j in range(len(ax)):
    ax[j].set_xlabel('Cycle number')
    ax[j].set_xlim([-5, 505])
    ax[j].set_title(chr(97 + j), loc="left", weight="bold")

fig.tight_layout()

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "knee_definition.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "knee_definition.eps", format="eps")


# KNEE IDENTIFICATION METHODS ================== FIGURE 4
m = pd.read_csv(path / 'severson2019_b2c30_health_fig04.csv')

t = np.array(m['cycs']).reshape((m['cycs'].shape[0],1))
q = np.array(m['q']).reshape((m['cycs'].shape[0],1))
dqdv = np.array(m['dqdv']).reshape((m['dqdv'].shape[0],1))

fig, ax = plt.subplots(2,3,figsize=(config.FIG_WIDTH*2,config.FIG_HEIGHT*1.5),
                       sharex=True, sharey=True)
t = np.array(m['cycs']).reshape((m['cycs'].shape[0],1))
q = np.array(m['q']).reshape((m['cycs'].shape[0],1))
IC = np.array(m['dqdv']).reshape((m['cycs'].shape[0],1))

colours = [blue, jade, claret, red, black, gold, grey]

# methods = ['Bacon-Watts','Kneedle','Diao et al.','Zhang et al.','Bisector','Comparison']
methods = ['Bacon-Watts','Kneedle','Diao et al.','Zhang et al.','Bisector',' ']

# This line will be changed once everything else is sorted
ref_nums = ['15','32','14','33','34','']

# the below code is all the stuff that applies to all plots. Yes, it is ugly.
n = 0;
for j in range(ax.shape[0]):
    for jj in range(ax.shape[1]):
        ax[j,jj].set_title(chr(97 + n), loc="left", weight="bold")
        if n<5:
            ax[j,jj].text(0,76,methods[n]+'$^{'+ref_nums[n]+'}$',color=colours[n])
        n = n+1
        
        if n > 3:
            ax[j,jj].set_xlabel('Cycle number')
        
        if jj in [0, 3]:
            ax[j,jj].set_ylabel('Capacity retention (%)')

# Below are the actual calculations
TK = np.zeros((6,2))
# Bacon-Watts
TK[0,0],TK[0,1] = bacon_watts_knee(ax[0,0],t,q,colours[0])
# Kneedle
TK[1,0],TK[1,1] = kneedle_identification(ax[0,1],t,q,colours[1])
# Diao et al.
TK[2,0],TK[2,1] = diao_knee(ax[0,2],t,q,colours[2])
# Zhang et al.
TK[3,0],TK[3,1] = zhang_knee(ax[1,0],t,q,dqdv,colours[3])
# Bisector
TK[4,0],TK[4,1] = knee_point_identification(ax[1,1],t,q,colours[4]);
# second derivative
TK[5,0] = 435; TK[5,1] = 88;

# plot capacity
ax[1,2].plot(t,q,color=colours[6])
# add in the knee estimates as a coloured scatter with vertical line for those 
# incapable of zooming a pdf
for j in range(6):   
    ax[1,2].scatter(TK[j,0],TK[j,1],color=colours[j],marker='x')
    ax[1,2].plot(np.array([TK[j,0],TK[j,0]]),np.array([80,98]),
                '--',color=colours[j])
    
# add in a zoomed in version into the bottom right plot
# new axes
ax_in = ax[1,2].inset_axes([10,80,310,12], transform=ax[1,2].transData)
# capacity curve
ax_in.plot(t,q,color=colours[6])
# add in the knee estimates
for j in range(6):
    ax_in.scatter(TK[j,0],TK[j,1],color=colours[j],marker='x')
    ax_in.plot(np.array([TK[j,0],TK[j,0]]),np.array([80,98]),
                '--',color=colours[j])
ax_in.set_yticks([])
ax_in.set_xlim([360, 445])
ax_in.set_ylim([85, 95])
    
# completely unnecessary if you are the kind of wizard who doesn't sanity 
# check your results
D = {'cycles':TK[:,0], 
      'capacity':TK[:,1]
     }

TK = pd.DataFrame(data=D)

print(TK)

fig.tight_layout() # adds nice whitespace

# Save figure as both .PNG and .EPS
fig.savefig(config.FIG_PATH / "knee_identification_methods.png", format="png", dpi=300)
fig.savefig(config.FIG_PATH / "knee_identification_methods.eps", format="eps")