
# coding: utf-8

# In[1]:


import scipy.io
import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


plt.rcParams['figure.figsize'] = 5, 5

# from IPython.core.display import HTML



# # What problem do HMMS solve?
# # Given a sequence of observations y find the probability of a state
# # $$P(x_t|y_{1:t}) = P(x_t|y_1,y_2,...,y_t)$$
#
# # ##### example 1.
# https://www.youtube.com/watch?v=jY2E6ExLxawa robot dog observes actions such as facebook, game of thrones watching, farting, and sleeping. what is the probability given a sequence from the above observation that the owner is sad or happy?
# # ##### example 2.
# weather (explain)
# # ##### example 3
# predict volatility tomorow given prices today
# # ##### example 4
# given observed video frames. find object's locationwe derive a recursion to compute
# # $$P(x_t|y_{1:t}) = P(x_t|y_1,y_2,...,y_t)$$
# assuming that we have as input
# # $$P(x_{t-1}|y_{1:t-1})$$
#
# # the recursion has two steps:
# # 1. prediction
# # 2. bayesian update
# we start of with
# # $$P(x_0)$$
# $$y_1 \rightarrow  P(x_1|y_{1})$$
# $$y_2 \rightarrow  P(x_2|y_{1:2})$$
# $$y_3 \rightarrow  P(x_3|y_{1:3})$$
# $$\vdots $$
# $$y_t \rightarrow  P(x_t|y_{1:t})$$
# everytime we get new information we compute the posterior.each of the equations above is a table. that says how sad or how happy you are.
# #### prediction

# $$P(x_t|y_{1:t-1}) =\sum_{x_{t-1}\epsilon \left \{  H,S \right \}}P(x_t,x_{t-1}|y_{1:t-1})$$

# $$P(x_t|y_{1:t-1}) =\sum_{x_{t-1}\epsilon \left \{  H,S \right \}}P(x_t|x_{t-1},y_{1:t-1})P(x_{t-1}|y_{1:t-1})$$
# due to the markov blanker x_t does not depend upon y_t-1 therefore
# $$P(x_t|x_{t-1},y_{1:t-1}) \rightarrow P(x_t|x_{t-1})$$

# $$P(x_t|y_{1:t-1}) =\sum_{x_{t-1}\epsilon \left \{  H,S \right \}}P(x_t|x_{t-1})P(x_{t-1}|y_{1:t-1})$$

# ### bayes update

# $$P(x_t|y_{1:t}) = P(x_t|y_t,y_{1:t-1})\overset{bayes}{\rightarrow}P(x_t|y_t,y_{1:t-1})= \frac{P(y_t|x_t,y_{1:t-1})P(x_t|y_{1:t-1})}
# {\sum_{x_t}P(y_t|x_t,y_{1:t-1})P(x_t|y_{1:t-1})}$$
# when x_t is given does y_t depend on the past? No. because of the markov blanked. eqyation above becomes
# $$P(x_t|y_{1:t}) = P(x_t|y_t,y_{1:t-1})\overset{bayes}{\rightarrow}P(x_t|y_t,y_{1:t-1})= \frac{P(y_t|x_t,y_{1:t-1})P(x_t|y_{1:t-1})}
# {\sum_{x_t}P(y_t|x_t,y_{1:t-1})P(x_t|y_{1:t-1})} = \frac{P(y_t|x_t)P(x_t|y_{1:t-1})}
# {\sum_{x_t}P(y_t|x_t)P(x_t|y_{1:t-1})}$$

# ### Problem Sets

# In[5]:


mat = scipy.io.loadmat('./TΕL606_labnotes_3.mat')

data = mat['price_move']

l =  data.shape[0]
price_move = np.zeros(l)
for i, p in enumerate(data):
    price_move[i] = p[0]

ys = price_move

df = pd.DataFrame(price_move,columns=['price_move'])


# In[6]:


q = 0.9
            #  G    B
px = np.array([0.2,0.8])
            #    G   B
pxx = np.array([[.8,.2],
                [.2,.8]])
              #  -1   +1
pyx = np.array([[1-q, q],
                #G
                #B
                [q,1-q]])


# In[7]:


px


# In[8]:


pxx


# In[9]:


pyx


# #### change the indexing to point to position in array in regards to stock movement

# In[10]:


ytra = []
for i in ys:
    if i==-1:
        ytra.append(0)
    else:
        ytra.append(1)

# 0 = good
# 1 = bad
# ### Forward-Backward Algorythm

# In[11]:


hidden_states=[0,1]


# ### Forward

# $$a_1(x_1)=P(x_1,y_1)=P(y_1|x_1)P(x_1)$$
# $$a_1(x_1=GOOD)=P(x_1=GOOD,y_1=-1)=P(y_1=-1|x_1=GOOD)P(x_1=GOOD)$$
# $$a_1(x_1=BAD)=P(x_1=BAD,y_1=-1)=P(y_1=-1|x_1=BAD)P(x_1=BAD)$$

# In[12]:


from utilities import  ForwardBackwardAlgorythm


# In[21]:


fba = ForwardBackwardAlgorythm(px,pxx,pyx,ytra)
forward = fba.forward()
backward = fba.backward()
gammas = fba.gammas()


# In[15]:


a1g = pyx[ytra[0],0]*px[0]#(1-q) * 0.2
a1b = pyx[ytra[0],1]*px[1]#q * .8
ai_ar = np.zeros((39, 2))
ai_ar[0, :] = [a1b, a1g]


# $$a_i(x_i)$$

# In[16]:


for t in range(1, 39):
    for xi in hidden_states:
        ai_ar[t, xi] = (ai_ar[t-1, 0]* pxx[0, xi] * pyx[0, ytra[t]]) + (ai_ar[t-1, 1] * pxx[1, xi] * pyx[1, ytra[t]])

# ai_ar
# $$b_i(x_i)$$

# In[17]:


b1g = 1
b1b = 1
bi_ar = np.zeros((39, 2))
bi_ar[-1,:] = [b1b, b1g]


# In[18]:


for t in np.arange(37,-1,-1):
    for xi in hidden_states:
        bi_ar[t,xi] = (bi_ar[t+1, 0] * pxx[xi, 0] * pyx[0, ytra[t+1]]) + (bi_ar[t+1, 1] * pxx[xi, 1] * pyx[1, ytra[t+1]])

# bi_ar
# $$gamma_i(x_i)$$

# In[19]:


l = []
for t in range(39):
    ab = (ai_ar[t] * bi_ar[t])
    s = ab.sum()
    d = ab/s
    l.append(d)

out = np.zeros((39,2))
for i, a in enumerate(l):
    out[i,:] = a


# # Plot stock movements

# In[20]:


df.index=range(1,40)
df.plot()
plt.plot(range(1,40),out[0:,0],color='r',label=r'P(GOOD|y)')
plt.legend(loc=4)
plt.xticks(range(1,40));
plt.grid()
plt.show()

# In[ ]:






### Actualdf.cumsum().plot()
# # ORIGINAL
import numpy as np
import time

start = ['Rain','Sun']
# state probabilities
p_start = [0.2,0.8]

#Transition probabilities
# These are calculated beforehand
t1 = ['Rain','Sun']
p_t1=[[0.4,0.6],
      [0.3,0.7]]

#t2 = [['W|R','Sh|R','C|R'],['W|Su','Sh|Su','C|Su']]
t2 = ['Walk','Shop','Clean']
#Emission probability
p_t2=[[0.1,0.4,0.5],
      [0.6,0.3,0.1]]

initial = np.random.choice(start,
                           replace=True,
                           p=p_start)

n = 4

st = 1
for i in range(n):
    if st:
        print('Setting the Initial State')
        state = initial
        st = 0
        print('Initial State:', state)
    if state == 'Rain':
        activity = np.random.choice(t2, p=p_t2[0])
        print('The state was?:', state)
        print('Do this today?: ', activity)
        state = np.random.choice(t1, p=p_t1[0])
    elif state == 'Sun':
        activity = np.random.choice(t2, p=p_t2[1])
        print('The state was?:',state)
        print('Do this today?: ', activity)
        state = np.random.choice(t1, p=p_t1[1])
    print("\n")
#     time.sleep(0.5)

    # Output (I printed out the hidden state too)
    # R R Shop -- R Clean -- Su Walk -- Su Walk -- Su Walk -- Su Clean -- Su Walk -- R Shop -- R Shop -- R Shop -- R Shop -- Su Shop -- R Clean -- Su Walk -- Su Walk -- R Shop -- R Clean -- R Clean -- Su Shop -- Su Shop
    import numpy as np
import time

start = ['BAD','GOOD']
# state probabilities
p_start = [0.8, 0.2]

#Transition probabilities
# These are calculated beforehand
t1 = ['BAD','GOOD']

p_t1=[[0.8,0.2],
      [0.2,0.8]]

#t2 = [['W|R','Sh|R','C|R'],['W|Su','Sh|Su','C|Su']]
t2 = [1,-1]
#Emission probability
p_t2=[[0.3,0.7],
      [0.7,0.3]]

initial = np.random.choice(start,
                           replace=True,
                           p=p_start)

n = 39

st = 1
activities  = []
for i in range(n):
    if st:
        print('Setting the Initial State')
        state = initial
        st = 0
        print('Initial State:', state)
    if state == 'BAD':
        activity = np.random.choice(t2, p=p_t2[0])
        print('The state was?:',state)
        print('Do this today?: ', activity)
        activities.append(activity)
        state = np.random.choice(t1, p=p_t1[0])
    elif state == 'GOOD':
        activity = np.random.choice(t2, p=p_t2[1])
        print('The state was?:',state)
        print('Do this today?: ', activity)
        activities.append(activity)
        state = np.random.choice(t1, p=p_t1[1])
    print("\n")df[0].valuesorig = df[0].values
origactivitiesplt.plot(orig, color='b')
plt.plot(activities, color='r')
# df.plot()
plt.yticks([-1,+1])len(activities)len(orig)Q=0
const = px[Q]#*pxx[Q,Q]
a10 = [const]
for i in range(1,39):
    O = ytra[:i]
    product = np.zeros(len(O)+1)
    product[0]=const
    for i, o in enumerate(O,start=1):
#         print(o)
        product[i] = pyx[o,Q]
    a10.append(product.prod())



Q=1
const = px[Q]#*pxx[Q,Q]
a11 = [const]
for i in range(1,39):
    O = ytra[:i]
    product = np.zeros(len(O)+1)
    product[0]=const
    for i, o in enumerate(O,start=1):
#         print(o)
        product[i] = pyx[o,Q]
    a11.append(product.prod())

ai = np.array(list(zip(a10,a11)))
