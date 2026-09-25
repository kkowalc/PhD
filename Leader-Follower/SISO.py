import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib        as mpl
import networkx as nx
import seaborn as sns
from scipy.integrate import solve_ivp
sns.set_theme(style="whitegrid", palette="colorblind",font_scale=3) 


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

f_l=lambda t,x: np.sin(t) #Leader dynamics
f_i=lambda x:2+1/2*np.cos(x)+1/4*np.cos(2*x) #Followers dynamics

#Estimation parameters
Kernel_fun=lambda x :1*(np.abs(x)<=1/2).astype(float) #Kernel
N=1000
a=0
b=2
sigma=0.1 #Disturbance - dispertion
h=0.1 #Bandwidth
L=1 #Lipschitz constant
delta=0.01 #Confidence parameter



k=np.array([1.,1.,1.,1.])
k=5*k #Consensus feedback gain

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
    return zeta_x

def e_x(x): #Tracking error for each follower
    e_x=np.zeros(2*M)
    for i in range(M):
        e_x[i]=x[i+1]-x[0]
    return np.array(e_x)


def u_x(x,k):
    return zeta_x(x)[k]

def ODE(t,x):
    dxdt=np.zeros((M+1))
    dxdt[0]=f_l(t,x[0])
    for i in range(M):
        dxdt[i+1]=-k[i]*u_x(x,i)+f_i(x[i+1])-est(x[i+1])


    return dxdt

def kernel_estimator( x_train, y_train, x, h,kernel_fun,L,sigma,delta):
       x_train = np.asarray(x_train)
       y_train = np.asarray(y_train)
       x=np.atleast_1d(x)
     

       np.seterr(divide='ignore', invalid='ignore')
       weights = kernel_fun((x[:, None]  - x_train[None, :] ) / h)
       num=np.sum(np.multiply(y_train, weights), axis=1)
   
       kappa=np.sum(weights,axis=1)
       NW_bound=np.inf*np.ones_like(x)
       y_hat=np.nan*np.ones_like(x)
       alpha = np.zeros_like(x)
       alpha[kappa <= 1] = np.sqrt(np.log(np.sqrt(2)*M*card/delta))
       alpha[kappa > 1] = np.sqrt(kappa[kappa > 1]*np.log(np.sqrt(1 + kappa[kappa > 1])*M*card/delta))
       
       
       NW_bound[kappa>0]= L*(h+rho) + 2*sigma*alpha[kappa>0]/kappa[kappa>0]

       y_hat[kappa>0]=num[kappa>0]/kappa[kappa>0]
       return y_hat,NW_bound

def est(x):
    idx = np.argmin(np.abs(x - x_est))
    f_hat=y_hat[idx]
    return f_hat

def plot_graph(adjacency_matrix, mylabels):
    gr = nx.from_numpy_array(adjacency_matrix)
    labels = {i: str(mylabels[i]) for i in range(len(mylabels))}
    nx.relabel_nodes(gr, labels, copy=False)

   # pos = nx.spring_layout(gr, seed=42, k=2.0)# or nx.kamada_kawai_layout(gr)
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
         arrowstyle="-",   # line only, no arrowhead (good for undirected graphs)
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
    
    
def random_connection_matrix(M,density):
    NoC=int(M*M*density)
    conn=np.random.randint(0, M, (2,NoC))
    MoC=np.zeros([M,M])
    MoC[conn[0],conn[1]]=1
    MoC[conn[1],conn[0]]=1
    np.fill_diagonal(MoC,0)
    return MoC



#Data samples 
exp=np.random.uniform(a,b,N)
eta=np.random.normal(0,sigma,N)
val=f_i(exp)+eta



x0=np.random.uniform(0,2,5) #Initial states
x0[0]=0 #leader state

k=np.array([1.,1.,1.,1.])
k=5*k
#values of time
T=20 #End time
tn=100000
t=np.linspace(0,T,tn)


#Model
card=21
x_est=np.linspace(a,b,card)
rho=np.abs(x_est[0]-x_est[1])/2
y_hat, b= kernel_estimator( exp, val, x_est, h,Kernel_fun,L,sigma,delta)

x=solve_ivp(ODE,[0, T],x0,t_eval=t)




#Plots

x_plot=np.linspace(a,b,1000)

AL= np.array([[0, 1, 0, 0, 0],
              [1, 0, 1, 1, 0],
              [0, 1, 0, 0, 1],
              [0, 1, 0, 0, 1],
              [0, 0, 1, 1, 0]])

plot_graph(AL,["\$1$","\$2$","\$3$","\$4$","\$l$"])
plot_graph(A,["\$1$","\$2$","\$3$","\$4$"])

plt.figure()

plt.plot(t,x.y[1,:],c="C4",lw=2,label="\$x_1$") 
plt.plot(t,x.y[2,:],c="C2",lw=2,label="\$x_2$") 
plt.plot(t,x.y[3,:],c="C1",lw=2,label="\$x_3$") 
plt.plot(t,x.y[4,:],"--",c="C0",lw=2,label="\$x_4$") 
plt.plot(t,x.y[0,:],c="black",lw=2,label="\$x_l$") 
plt.xlabel("time") 
plt.xlim([0,T])
plt.ylim([-0.01,4])
plt.show()
plt.grid(True)
plt.legend(ncol=5,columnspacing=1)

ne=np.zeros(len(x.t))
for n in range(0,len(x.t)):
    ne[n]=np.linalg.norm(e_x(x.y[:,n]),2)
v=4*(1+np.max(b))**2
r=np.sqrt(np.max(np.linalg.eigvalsh(TL))/np.min(np.linalg.eigvalsh(TL)))*2*np.sqrt(v)/(np.sqrt(3)*np.min(k)*np.min(np.linalg.eigvalsh(TL)))

plt.figure(5)
plt.plot(t,ne,lw=2, color="C1")
plt.xlim([0,T])
plt.grid(True)
plt.xlabel("time") 
plt.hlines(r, 0, T, color="C1", linestyle="--")
