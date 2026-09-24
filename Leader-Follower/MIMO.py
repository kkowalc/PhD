import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib        as mpl
import networkx as nx
import seaborn as sns
from scipy.integrate import solve_ivp
sns.set_theme(style="whitegrid", palette="colorblind") 


mpl.rcParams['svg.fonttype'] = 'none'

np.random.seed(2026)

plt.close("all")


#Network parameters
M=4 #Number of followers

A=np.array([[0, 1, 0, 0],   #Adjacency matrix
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0]])

D = np.diag(np.sum(A, axis=1)) #Degree matrix
B=np.zeros([4,4])   #Leader connection matrix
B[2,2]=1
B[3,3]=1
L=D-A #Laplacian
TL=L+B #Augmented Laplacian

#Dynamics
f_l1=lambda t,x1,x2: np.sin(t)-x1 #Leader dynamics
f_l2=lambda t,x1,x2: np.cos(t)-x2

f_i1=lambda x1,x2: x2*np.cos(x1) #Followers dynamics
f_i2=lambda x1,x2: x1*np.sin(x2)


#Estimation parameters
Kernel_fun=lambda x :1*(np.abs(x)<=1/2).astype(float) #Kernel
N=5000 #Number of data samples
sigma=0.1 #Disturbance - dispertion
h=0.5 #Bandwidth
L=1 #Lipschitz constant
delta=0.01  #Confidence parameter
a=-1
b=1

global_domain=np.array([[a,a],[b,b]]) #state space
gs=0.25
grid_step=np.array([gs,gs]) # grid step for every estimation point
axes = [
    np.arange(global_domain[0, i],
              global_domain[1, i] + grid_step[i],
              grid_step[i])
    for i in range(global_domain.shape[1])
]

mesh = np.meshgrid(*axes, indexing='ij')
grid = np.column_stack([m.ravel() for m in mesh])
mesh=np.array(mesh)




def A_zeta(i,A,x,d):    #follower-follower consensus error
    xi=np.sum(A[i,:])*x[i+1+d*(M+1)]
    
    for j in range(len(A[i,:])):
        xi=xi-A[i,j]*x[j+1+d*(M+1)]
    return xi

def B_zeta(i,A,x,d):    #leader-follower consensus error
    xi=0
    if B[i,i]==1:
        xi=x[i+1+d*(M+1)]-x[d*(M+1)]
    return xi


def zeta_x(x):  #Consensus error for each follower
    zeta_x=np.zeros(2*M)
    for i in range(M):
        zeta_x[i]=A_zeta(i,A,x,0)+B_zeta(i,B,x,0)
        zeta_x[i+M]=A_zeta(i,A,x,1)+B_zeta(i,B,x,1)
    return zeta_x

def e_x(x): #Tracking error for each follower
    e_x=np.zeros(2*M)
    for i in range(M):
        e_x[i]=x[i+1]-x[0]
        e_x[i+M]=x[i+1+M]-x[5]
    return np.array(e_x)

def u_x(x,k):
    return zeta_x(x)[k]

def ODE(t,x):   
    dxdt=np.zeros(2*(M+1))
    
    dxdt[0]=f_l1(t,x[0],x[5])
    dxdt[5]=f_l2(t,x[0],x[5])
    
    for i in range(M):
        dxdt[i+1]=-k[i]*u_x(x,i)+f_i1(x[i+1],x[i+2+M])-est1(x[i+1],x[i+2+M])
        dxdt[i+2+M]=-k[i]*u_x(x,i+M)+f_i2(x[i+1],x[i+2+M])-est2(x[i+1],x[i+2+M])
    

    return dxdt

def kernel_estimator( x_train, y_train, x, h,kernel_fun,L,sigma,delta):
       x_train = np.asarray(x_train)
       y_train = np.asarray(y_train)
       x = np.asarray(x)
       np.seterr(divide='ignore', invalid='ignore')
       distance=np.linalg.norm(x[:, None, :] - x_train[None, :, :], axis=2)
       weights = kernel_fun(distance/h)
       num=weights @ y_train
         
       kappa=np.sum(weights, axis=x.ndim-1)
       NW_bound=np.inf*np.ones(x.shape[0])
       y_hat=np.nan*np.ones([x.shape[0],y_train.shape[1]])
       alpha = np.zeros(x.shape[0])
       alpha[kappa <= 1] = np.sqrt(np.log(np.sqrt(2)/delta))
       alpha[kappa > 1] = np.sqrt(kappa[kappa > 1]*np.log(np.sqrt(1 + kappa[kappa > 1])*4*11/delta))
       
       
       NW_bound[kappa>0]= L*(h+0.3) + 2*sigma*alpha[kappa>0]/kappa[kappa>0]
       y_hat=num/kappa[:, None]
       return y_hat,NW_bound

def est1(x1,x2):
    x = np.column_stack((x1, x2))
    idx = np.argmin(np.linalg.norm(x - x_est, axis=1))
    f_hat=y_hat1[idx]
    return f_hat

def est2(x1,x2):
    x = np.column_stack((x1, x2))
    idx = np.argmin(np.linalg.norm(x - x_est, axis=1))
    f_hat=y_hat2[idx]
    return f_hat

def plot_graph(adjacency_matrix, mylabels):
    gr = nx.from_numpy_array(adjacency_matrix)
    labels = {i: str(mylabels[i]) for i in range(len(mylabels))}
    nx.relabel_nodes(gr, labels, copy=False)

    pos=nx.spring_layout(gr, seed=42, k=3.0)
    fig, ax = plt.subplots(figsize=(11, 9), facecolor="white")
    ax.set_facecolor("#0f172a")

    nx.draw_networkx_edges(
         gr,
         pos,
         ax=ax,
         edge_color="#334155",
         width=2,
         alpha=0.65,
         connectionstyle="arc3,rad=0.0",
         arrows=True,
         arrowstyle="-", 
     )
    nx.draw_networkx_nodes(
        gr, pos, ax=ax,
        node_color="#2190e3ff", node_size=3000,
        edgecolors="#e2e8f0", linewidths=2,
    )
    nx.draw_networkx_labels(
        gr, pos, ax=ax,
        font_size=12, font_weight="bold", font_color="black",
    )

    ax.axis("off")
    plt.tight_layout()
    plt.show()


#Data samples 
exp=np.random.uniform(a,b,(N,2))
eta1=np.random.normal(0,sigma,N)
eta2=np.random.normal(0,sigma,N)
val1=f_i1(exp[:,0],exp[:,1])+eta1
val2=f_i2(exp[:,0],exp[:,1])+eta2
val = np.column_stack((val1, val2))


#Model
x_est=grid #estimation points
y_hat,beta= kernel_estimator( exp, val, x_est, h,Kernel_fun,L,sigma,delta)
y_hat1=y_hat[:,0]
y_hat2=y_hat[:,1]


x0=np.random.uniform(a,b,10)    #Initial states

#leader state
x0[0]=0
x0[5]=0

k=np.array([1.,1.,1.,1.])
k=10*k #consensus feedback gain

T=10 #End time
tn=5000
t=np.linspace(0,T,tn)



x=solve_ivp(ODE,[0, T],x0,t_eval=t)



#Plots
x_plot = np.linspace(a, b, 100)
y_plot = np.linspace(a, b, 100)

X_plot, Y_plot = np.meshgrid(x_plot, y_plot)

plot_x = np.column_stack([
    X_plot.ravel(),
    Y_plot.ravel()
])


AL= np.array([[0, 1, 0, 0, 0],
              [1, 0, 1, 1, 0],
              [0, 1, 0, 0, 1],
              [0, 1, 0, 0, 1],
              [0, 0, 1, 1, 0]])

plot_graph(AL,["$1$","$2$","$3$","$4$","$l$"])
plot_graph(A,["$1$","$2$","$3$","$4$"])

K = int(np.sqrt(len(grid)))

X = grid[:, 0].reshape(K, K)
Y = grid[:, 1].reshape(K, K)

F1 = y_hat1.reshape(K, K)
F2=     y_hat2.reshape(K, K)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x.y[1,:],x.y[6,:],x.t,c="C4",lw=2,label=r"$x_1$") 
ax.plot(x.y[2,:],x.y[7,:],x.t,c="C2",lw=2,label="$x_2$") 
ax.plot(x.y[3,:],x.y[8,:],x.t,c="C1",lw=2,label="$x_3$") 
ax.plot(x.y[4,:],x.y[9,:],x.t,c="C0",lw=2,label="$x_4$") 
ax.plot(x.y[0,:],x.y[5,:],x.t,c="black",lw=2,label="$x_l$") 
ax.set_zlabel('time', labelpad=15, rotation=90)

ax.view_init(elev=20, azim=-150)
ax.legend(ncol=5)

ne=np.zeros(len(x.t))
for n in range(0,len(x.t)):
    ne[n]=np.linalg.norm(e_x(x.y[:,n]),2)
v=4*(1+np.max(beta))**2
r=np.sqrt(np.max(np.linalg.eigvalsh(TL))/np.min(np.linalg.eigvalsh(TL)))*2*np.sqrt(v)/(np.sqrt(3)*np.min(k)*np.min(np.linalg.eigvalsh(TL)))


plt.figure()
plt.plot(t,ne,label="$|| e ||_2$")
plt.xlabel("time")
plt.legend()
