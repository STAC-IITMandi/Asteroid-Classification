import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import moid as md


# attributes = [ 'Eccentricity', 'Semi Major Axis', 'Perihelion Distance', 'Inclination', 'Asc Node Longitude', 'Perihelion Arg', 'Minimum Orbit Intersection']
attributes = ['a', 'e', 'i', 'w', 'om', 'q', 'H', 'neo', 'pha', 'moid']

colnames = {'a':  "Semi-major axis, AU",
            'q':  "Perihelion distance, AU",
            'i':  "Inclination, deg",
            'e':  "Eccentricity",
            'w':  "Argument of perihelion, deg",
            'om': "Longitude of the ascending node, deg"}

database=pd.read_csv("latest_fulldb.csv",usecols=attributes)
neas=database[database["neo"] == "Y"]
neas=database.dropna(subset=[ 'e', 'a', 'q', 'i', 'om', 'w', 'moid'])
# neas=database.dropna(subset=[ 'Eccentricity', 'Semi Major Axis', 'Perihelion Distance', 'Inclination', 'Asc Node Longitude', 'Perihelion Arg', 'Minimum Orbit Intersection'])

# print (len(neas[neas.a > 4]))
# print (len(neas[neas.i > 90]))

neas.reset_index(inplace=True,drop=True)

neas = neas[neas['a'] < 4]
neas = neas[neas['i'] < 90]
neas_close = neas[neas['q'] <= 1.08]
neas_close.reset_index(inplace=True,drop=True)
neas_close.to_csv('original.csv',index=False)
# moid_lis=[]
# for i in range(len(neas_close)): 
#     row_acc=neas_close.iloc[i]
#     w_, i_, om_ = np.radians([row_acc["w"],row_acc['i'],row_acc['om']])
#     moid_row=md.get_moid(row_acc['a'],row_acc['e'],w_,i_,om_,)
#     moid_lis.append(moid_row)
# neas_close['moid_new']=moid_lis

haz_real=neas_close[neas_close["moid"] <= 0.05]
nohaz_real = neas_close[neas_close["moid"] > 0.05]

print ("Number of real PHAs", len(haz_real))
print ("Number of real NHAs:", len(nohaz_real))


fig = plt.figure()
ax = fig.add_subplot(111)

x_axis_1=haz_real['w']
x_axis_2=nohaz_real['w']

y_axis_1=haz_real['q']
y_axis_2=nohaz_real['q']

plt.scatter(x_axis_1,y_axis_1,c="orange",s=1,alpha=1)
plt.scatter(x_axis_2,y_axis_2,c="blue",s=1,alpha=1)
plt.gca().invert_yaxis()
plt.xlabel("Argument of perihelion, deg")
plt.ylabel("Perihelion distance, AU")
plt.title("Original data")
plt.show()
plt.clf()

# for "a" rayleigh distribution
# params = stats.rayleigh.fit(neas_close["a"].values)
# rayleigh_dist = stats.rayleigh(*params)
# random_samples = rayleigh_dist.rvs(size=10) 
# print("Fitted Parameters:", params)
# print("Random Samples:", random_samples)

# plt.hist(neas_close["a"].values, bins=10,density=True, alpha=0.6,color='g', label='Data') 
# x = np.linspace(0, neas_close["a"].values, 100)
# plt.plot(x, rayleigh_dist.pdf(x), alpha=0.6)
# plt.title('Fitting Data to Rayleigh Distribution')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# # plt.show()

#for "i" lognorm distribution
# params = stats.lognorm.fit(neas_close["i"].values)
# lognorm_dist = stats.lognorm(*params)
# random_samples = lognorm_dist.rvs(size=10) 
# print("Fitted Parameters:", params)
# print("Random Samples:", random_samples)

# plt.hist(neas_close["i"].values, bins=10,density=True, alpha=0.6,color='g', label='Data') 
# x = np.linspace(0, neas_close["i"].values, 100)
# plt.plot(x, lognorm_dist.pdf(x), alpha=0.6)
# plt.title('Fitting Data to lognorm Distribution')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.show()




#for w uniform

# params = stats.uniform.fit(neas_close["w"].values) #rtrn loc,scale
# uniform_dist = stats.uniform(*params)
# random_samples = uniform_dist.rvs(size=10)
# print("Fitted Parameters:", params)
# print("Random Samples:", random_samples)

# def harmonic_pdf(x, A, omega):
#     return A * np.cos(omega * x) / (2 * np.pi)

# params, _ = curve_fit(harmonic_pdf, neas_close["om"], np.ones_like(neas_close["om"]))
# A_fit, omega_fit = params
# print("Estimated Parameters :", A_fit, omega_fit)
# random_samples = np.random.uniform(0, 2*np.pi, 10)  # Generate uniform random samples
# fitted_samples = harmonic_pdf(random_samples, A_fit, omega_fit)
# print(fitted_samples)
# params, _ = curve_fit(harmonic_pdf, neas_close["om"], np.ones_like(neas_close["om"]))
# a, b = params
# random_samples = np.random.uniform(a, b, size=10) 

# print("Random Samples:", random_samples)

# Plot the original data and the fitted Harmonic distribution
# plt.hist(neas_close["om"], bins=10, density=True, alpha=0.6, color='g', label='Data')
# x = np.linspace(a, b, 100)
# plt.plot(x, harmonic_pdf(x, a, b), label='Fitted Harmonic PDF')
# plt.legend(loc='best')
# plt.title('Fitting Data to Harmonic Distribution')
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.show()

#for "i" johnsonsu distribution
# params = stats.johnsonsu.fit(neas_close["q"].values)
# johnsonsu_dist = stats.johnsonsu(*params)
# random_samples = johnsonsu_dist.rvs(size=10) 
# print("Fitted Parameters:", params)
# print("Random Samples:", random_samples)

# plt.hist(neas_close["q"].values, bins=10,density=True, alpha=0.6,color='g', label='Data') 
# x = np.linspace(0, neas_close["q"].values, 100)
# plt.plot(x, johnsonsu_dist.pdf(x), alpha=0.6)
# plt.xlabel('Value')
# plt.ylabel('Probability Density')
# plt.show()
# for i in range(50):
#     strow=neas_close.iloc[i]
#     print("moid",md.Nget_moid(strow['a'],strow['e'],strow['w'],strow['i'],strow['om']))
#     print('ori moid',strow['moid'])
params = stats.uniform.fit(neas_close["a"].values) 
uniform_dist1 = stats.uniform(*params)

params = stats.uniform.fit(neas_close["i"].values) 
uniform_dist2 = stats.uniform(*params)

params = stats.uniform.fit(neas_close["w"].values) 
uniform_dist3 = stats.uniform(*params)

params = stats.uniform.fit(neas_close["om"].values) 
uniform_dist4 = stats.uniform(*params)

params = stats.uniform.fit(neas_close["q"].values) 
uniform_dist5 = stats.uniform(*params)

# params = stats.uniform.fit(neas_close["Eccentricity"].values) 
# uniform_dist6 = stats.uniform(*params)

uniform_dataset=[]
counter=0
while(counter<1000):
    # a_gen= uniform_dist1.rvs(size=1)[0]
    a_gen= np.random.uniform(neas_close["a"].min(),neas_close["a"].max(),1)[0]
    i_gen= np.random.uniform(neas_close["i"].min(),neas_close["i"].max(),1)[0]
    w_gen= np.random.uniform(neas_close["w"].min(),neas_close["w"].max(),1)[0]
    om_gen= np.random.uniform(neas_close["om"].min(),neas_close["om"].max(),1)[0]
    q_gen= np.random.uniform(neas_close["q"].min(),neas_close["q"].max(),1)[0]
    # i_gen= uniform_dist2.rvs(size=1)[0]
    # w_gen= uniform_dist3.rvs(size=1)[0]
    # om_gen= uniform_dist4.rvs(size=1)[0]
    # q_gen= uniform_dist5.rvs(size=1)[0]
    e_gen= (a_gen-q_gen)/a_gen
    if(e_gen>1):
        e_gen=0.99
    lis=[a_gen,i_gen,w_gen,om_gen,q_gen,e_gen]
    if(q_gen<1.3 and e_gen>0):
        uniform_dataset.append(lis)
        counter+=1

uniform_df=pd.DataFrame(uniform_dataset,columns=['a', 'i', 'w', 'om', 'q','e'])

moid_lis=[]
for i in range(len(uniform_df)):
    row_acc=uniform_df.iloc[i]
    w_, i_, om_ = np.radians([row_acc["w"],row_acc['i'],row_acc['om']])
    moid_row=md.get_moid(row_acc['a'],row_acc['e'],w_, i_, om_ )
    moid_lis.append(moid_row)

uniform_df['moid']=moid_lis
haz_virtual=uniform_df[uniform_df["moid"] <= 0.05]
nohaz_virtual = uniform_df[uniform_df["moid"] > 0.05]


x_axis_1=haz_virtual["w"]
x_axis_2=nohaz_virtual["w"]

y_axis_1=haz_virtual["q"]
y_axis_2=nohaz_virtual["q"]

plt.scatter(x_axis_1,y_axis_1,c="orange",s=1,alpha=1)
plt.scatter(x_axis_2,y_axis_2,c="blue",s=1,alpha=1)
plt.gca().invert_yaxis()
plt.xlabel("Argument of perihelion, deg")
plt.ylabel("Perihelion distance, AU")
plt.title("uniform")
plt.show()
plt.clf()

params = stats.rayleigh.fit(neas_close["a"].values)
rayleigh_dist = stats.rayleigh(*params)

params = stats.lognorm.fit(neas_close["i"].values)
lognorm_dist = stats.lognorm(*params)

params = stats.uniform.fit(neas_close["w"].values) 
uniform_dist = stats.uniform(*params)

def harmonic_pdf(x, A, omega):
    return A * np.cos(omega * x) / (2 * np.pi)

params, _ = curve_fit(harmonic_pdf, neas_close["om"], np.ones_like(neas_close["om"]))
A_fit, omega_fit = params


params = stats.johnsonsu.fit(neas_close["q"].values)
johnsonsu_dist = stats.johnsonsu(*params)


nonuniform_dataset=[]
counter=0
while(counter<200000):
    a_gen= rayleigh_dist.rvs(size=1)[0]
    i_gen= lognorm_dist.rvs(size=1)[0]
    w_gen= uniform_dist.rvs(size=1)[0]
    random_sample = np.random.uniform(0, 2*np.pi, 1)
    om_gen = harmonic_pdf(random_sample, A_fit, omega_fit)[0]
    q_gen= johnsonsu_dist.rvs(size=1)[0]
    e_gen= (a_gen-q_gen)/a_gen
    lis=[a_gen,i_gen,w_gen,om_gen,q_gen,e_gen]
    if(e_gen>1):
        e_gen=0.99
    if(q_gen<1.3 and e_gen>0):
        nonuniform_dataset.append(lis)
        counter+=1

nonuniform_df=pd.DataFrame(nonuniform_dataset,columns=['a', 'i', 'w', 'om', 'q','e'])

moid_lis=[]
for i in range(len(nonuniform_df)):
    row_acc=nonuniform_df.iloc[i]
    w_, i_, om_ = np.radians([row_acc["w"],row_acc['i'],row_acc['om']])
    moid_row=md.get_moid(row_acc['a'],row_acc['e'],w_, i_, om_ )
    moid_lis.append(moid_row)

nonuniform_df['moid']=moid_lis

haz_virtual_non=nonuniform_df[nonuniform_df["moid"] <= 0.05]
nohaz_virtual_non = nonuniform_df[nonuniform_df["moid"] > 0.05]


x_axis_1=haz_virtual_non["w"]
x_axis_2=nohaz_virtual_non["w"]

y_axis_1=haz_virtual_non["q"]
y_axis_2=nohaz_virtual_non["q"]

plt.scatter(x_axis_1,y_axis_1,c="orange",s=0.25,alpha=1)
plt.scatter(x_axis_2,y_axis_2,c="blue",s=0.25,alpha=1)
plt.gca().invert_yaxis()
plt.xlabel("Argument of perihelion, deg")
plt.ylabel("Perihelion distance, AU")
plt.title("non uniform virtual")
plt.show()
plt.clf()

neas_close=neas_close[['a', 'i', 'w', 'om', 'q','e','moid']]
df_name=[neas_close,uniform_df,nonuniform_df]
final_df=pd.concat(df_name)
print(len(final_df))

pd.to_csv('final_generated.csv',header=False,index=False)

params = stats.rayleigh.fit(neas_close["a"].values)
rayleigh_dist = stats.rayleigh(*params)

params = stats.lognorm.fit(neas_close["i"].values)
lognorm_dist = stats.lognorm(*params)

params = stats.uniform.fit(neas_close["w"].values) 
uniform_dist = stats.uniform(*params)

def harmonic_pdf(x, A, omega):
    return A * np.cos(omega * x) / (2 * np.pi)

params, _ = curve_fit(harmonic_pdf, neas_close["om"], np.ones_like(neas_close["om"]))
A_fit, omega_fit = params


params = stats.johnsonsu.fit(neas_close["q"].values)
johnsonsu_dist = stats.johnsonsu(*params)


nonuniform_dataset=[]
counter=0
while(counter<1000):
    a_gen= rayleigh_dist.rvs(size=1)[0]
    while(neas_close['a'].min()>a_gen or a_gen>neas_close['a'].max()):
        a_gen= rayleigh_dist.rvs(size=1)[0]
    # print('h')
    
    
    i_gen= lognorm_dist.rvs(size=1)[0]
    while(neas_close['i'].min()>i_gen or i_gen>neas_close['i'].max()):
        i_gen= rayleigh_dist.rvs(size=1)[0]
    # print('h')
    
    
    w_gen= uniform_dist.rvs(size=1)[0]
    while(neas_close['w'].min()>w_gen or w_gen>neas_close['w'].max()):
        w_gen= rayleigh_dist.rvs(size=1)[0]
    # print('h')
   
    
    random_sample = np.random.uniform(0, 2*np.pi, 1)
    om_gen = harmonic_pdf(random_sample, A_fit, omega_fit)[0]
    # while(neas_close['om'].min()>om_gen or om_gen>neas_close['om'].max()):
    #     random_sample = np.random.uniform(0, 2*np.pi, 1)
    #     om_gen = harmonic_pdf(random_sample, A_fit, omega_fit)[0]
    #     print('a')
    # print('h')
    
    
    q_gen= johnsonsu_dist.rvs(size=1)[0]
    while(neas_close['q'].min()>q_gen or q_gen>neas_close['q'].max()):
        q_gen= rayleigh_dist.rvs(size=1)[0]
    # print('h')
   
    
    e_gen= (a_gen-q_gen)/a_gen
    lis=[a_gen,i_gen,w_gen,om_gen,q_gen,e_gen]
    if(e_gen>1):
        e_gen=0.99
    if(q_gen<1.3 and e_gen>0 ):
        nonuniform_dataset.append(lis)
        counter+=1

nonuniform_df=pd.DataFrame(nonuniform_dataset,columns=['a', 'i', 'w', 'om', 'q','e'])

moid_lis=[]
for i in range(len(nonuniform_df)):
    row_acc=nonuniform_df.iloc[i]
    w_, i_, om_ = np.radians([row_acc["w"],row_acc['i'],row_acc['om']])
    moid_row=md.get_moid(row_acc['a'],row_acc['e'],w_, i_, om_ )
    moid_lis.append(moid_row)

nonuniform_df['moid']=moid_lis

haz_virtual_non=nonuniform_df[nonuniform_df["moid"] <= 0.05]
nohaz_virtual_non = nonuniform_df[nonuniform_df["moid"] > 0.05]


x_axis_1=haz_virtual_non["w"]
x_axis_2=nohaz_virtual_non["w"]

y_axis_1=haz_virtual_non["q"]
y_axis_2=nohaz_virtual_non["q"]

plt.scatter(x_axis_1,y_axis_1,c="orange",s=0.25,alpha=1)
plt.scatter(x_axis_2,y_axis_2,c="blue",s=0.25,alpha=1)
plt.gca().invert_yaxis()
plt.xlabel("Argument of perihelion, deg")
plt.ylabel("Perihelion distance, AU")
plt.title("non uniform virtual")
plt.show()
plt.clf()

nonuniform_df.to_csv('final_generated.csv',header=False,index=False)