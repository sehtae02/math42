#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp


# In[2]:


os.chdir("/Users/stae/Desktop/Math_42/final_files")
os.getcwd()


# In[ ]:


data2010


# In[298]:


# read in data files
data2010 = pd.read_csv("ACS_10_5YR_DP02_with_ann.csv")
data2011 = pd.read_csv("ACS_11_5YR_DP02_with_ann.csv")
data2012 = pd.read_csv("ACS_12_5YR_DP02_with_ann.csv")
data2013 = pd.read_csv("ACS_13_5YR_DP02_with_ann.csv")
data2014 = pd.read_csv("ACS_14_5YR_DP02_with_ann.csv")
data2015 = pd.read_csv("ACS_15_5YR_DP02_with_ann.csv")
data2016 = pd.read_csv("ACS_16_5YR_DP02_with_ann.csv")
excel = pd.ExcelFile("MCM_NFLIS_Data.xlsx")
data_excel = excel.parse("Data")


# In[4]:


# create new data frame for each state from original excel file
oh = data_excel.loc[data_excel['State'] == "OH"].reset_index(drop = True)
ky = data_excel.loc[data_excel['State'] == "KY"].reset_index(drop = True)
pa = data_excel.loc[data_excel['State'] == "PA"].reset_index(drop = True)
wv = data_excel.loc[data_excel['State'] == "WV"].reset_index(drop = True)
va = data_excel.loc[data_excel['State'] == "VA"].reset_index(drop = True)


# In[5]:


# graph for drug reports by state 
plt.figure(figsize=(10, 8))
plt.plot(oh["YYYY"], oh["TotalDrugReportsState"], label = "OH")
plt.plot(ky["YYYY"], ky["TotalDrugReportsState"], label = "KY")
plt.plot(pa["YYYY"], pa["TotalDrugReportsState"], label = "PA")
plt.plot(wv["YYYY"], wv["TotalDrugReportsState"], label = "WV")
plt.plot(va["YYYY"], va["TotalDrugReportsState"], label = "VA")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Total Drug Reports", fontsize=14)
plt.title("State Total Drug Reports 2010-2017")
plt.grid(True)
plt.legend(title = "State", fontsize = 11)
plt.show()


# In[325]:


# Create array of top 5 counties for each state based on highest drug reports
oh_counties = oh.sort_values("TotalDrugReportsCounty", ascending=False)["COUNTY"].unique()#[0:5]
ky_counties = ky.sort_values("TotalDrugReportsCounty", ascending=False)["COUNTY"].unique()#[0:5]
pa_counties = pa.sort_values("TotalDrugReportsCounty", ascending=False)["COUNTY"].unique()[0:5]
wv_counties = wv.sort_values("TotalDrugReportsCounty", ascending=False)["COUNTY"].unique()[0:5]
va_counties = va.sort_values("TotalDrugReportsCounty", ascending=False)["COUNTY"].unique()[0:5]


# In[7]:


# Create arrays of drug reports by year for each county for Ohio
oh_c1 = oh[oh["COUNTY"] == oh_counties[0]]["TotalDrugReportsCounty"].unique()
oh_c2 = oh[oh["COUNTY"] == oh_counties[1]]["TotalDrugReportsCounty"].unique()
oh_c3 = oh[oh["COUNTY"] == oh_counties[2]]["TotalDrugReportsCounty"].unique()
oh_c4 = oh[oh["COUNTY"] == oh_counties[3]]["TotalDrugReportsCounty"].unique()
oh_c5 = oh[oh["COUNTY"] == oh_counties[4]]["TotalDrugReportsCounty"].unique()


# In[324]:


# Graph for drug reports by county for Ohio
plt.figure(figsize=(10,8))
plt.plot(oh["YYYY"].unique(), oh_c1, label = "Cuyahoga")
plt.plot(oh["YYYY"].unique(), oh_c2, label = "Hamilton")
plt.plot(oh["YYYY"].unique(), oh_c3, label = "Franklin")
plt.plot(oh["YYYY"].unique(), oh_c4, label = "Montgomery")
plt.plot(oh["YYYY"].unique(), oh_c5, label = "Lake")
plt.legend(title = "County")
plt.xlabel("Year")
plt.ylabel("Drug Reports")
plt.title("Top 5 Highest Counties in Ohio by Total Drug Reports 2010-2017")
plt.savefig("ohio_counties.jpg")
plt.show()


# In[331]:


plt.plot(oh["YYYY"], oh["TotalDrugReportsCounty"])


# In[9]:


# Create arrays of drug reports by year for each county for Kentucky
ky_c1 = ky[ky["COUNTY"] == ky_counties[0]]["TotalDrugReportsCounty"].unique()
ky_c2 = ky[ky["COUNTY"] == ky_counties[1]]["TotalDrugReportsCounty"].unique()
ky_c3 = ky[ky["COUNTY"] == ky_counties[2]]["TotalDrugReportsCounty"].unique()
ky_c4 = ky[ky["COUNTY"] == ky_counties[3]]["TotalDrugReportsCounty"].unique()
ky_c5 = ky[ky["COUNTY"] == ky_counties[4]]["TotalDrugReportsCounty"].unique()
ky_counties


# In[322]:


# Graph for drug reports by county for Kentucky
plt.figure(figsize=(10,8))
plt.plot(ky["YYYY"].unique(), ky_c1, label = "Jefferson")
plt.plot(ky["YYYY"].unique(), ky_c2, label = "Fayette")
plt.plot(ky["YYYY"].unique(), ky_c3, label = "Kenton")
plt.plot(ky["YYYY"].unique(), ky_c4, label = "Campbell")
plt.plot(ky["YYYY"].unique(), ky_c5, label = "Bell")
plt.legend(title = "County")
plt.xlabel("Year")
plt.ylabel("Drug Reports")
plt.title("Top 5 Highest Counties by in Kentucky Total Drug Reports 2010-2017")
plt.savefig("ky_counties.jpg")


# In[11]:


# subset 2010 data to only have geographical id and amount of households
households = pd.to_numeric(data2010["HC01_VC03"][1:,])
id_house = pd.to_numeric(data2010["GEO.id2"][1:,])
id_and_house = pd.concat([id_house, households], axis=1)
id_and_house = id_and_house.rename({"GEO.id2" : "id", "HC01_VC03": "households"}, axis = 1)
id_and_house


# In[12]:


# population for kentucky 2010 is 1,676,708
ky_households = id_and_house.loc[id_and_house["id"] < 30000]
ky_pop = ky_households["households"].sum()
ky_pop


# In[13]:


# population for ohio 2010 is 4,552,270
oh_households = id_and_house.loc[(id_and_house["id"] > 30000) & (id_and_house["id"] < 40000)]
oh_pop = oh_households["households"].sum()
oh_pop


# In[14]:


# population for pennsylvania 2010 is 4,940,581
pa_households = id_and_house.loc[(id_and_house["id"] > 40000) & (id_and_house["id"] < 50000)]
pa_pop = pa_households["households"].sum()
pa_pop


# In[15]:


# population for virginia 2010 is 4,552,270
va_households = id_and_house.loc[(id_and_house["id"] > 50000) & (id_and_house["id"] < 54000)]
va_pop = va_households["households"].sum()
va_pop


# In[16]:


# population for west virginia 2010 is 740,874
wv_households = id_and_house.loc[(id_and_house["id"] > 54000)]
wv_pop = wv_households["households"].sum()
wv_pop


# In[17]:


# subset 2015 data to only have geographical id and amount of households
households_15 = pd.to_numeric(data2015["HC01_VC03"][1:,])
id_house_15 = pd.to_numeric(data2015["GEO.id2"][1:,])
id_and_house_15 = pd.concat([id_house_15, households_15], axis=1)
id_and_house_15 = id_and_house_15.rename({"GEO.id2" : "id", "HC01_VC03": "households"}, axis = 1)
id_and_house_15

# subset 2016 data to only have geographical id and amount of households
households_16 = pd.to_numeric(data2016["HC01_VC03"][1:,])
id_house_16 = pd.to_numeric(data2016["GEO.id2"][1:,])
id_and_house_16 = pd.concat([id_house_16, households_16], axis=1)
id_and_house_16 = id_and_house_16.rename({"GEO.id2" : "id", "HC01_VC03": "households"}, axis = 1)
id_and_house_16


# In[18]:


# kentucky 2015: 1,708,499
ky_households_15 = id_and_house_15.loc[id_and_house_15["id"] < 30000]
ky_pop_15 = ky_households_15["households"].sum()
ky_pop_15


# In[19]:


# ohio 2015: 4,585,084
oh_households_15 = id_and_house_15.loc[(id_and_house_15["id"] > 30000) & (id_and_house_15["id"] < 40000)]
oh_pop_15 = oh_households_15["households"].sum()
oh_pop_15


# In[20]:


# population for pennsylvania 2015 is 4,958,859
pa_households_15 = id_and_house_15.loc[(id_and_house_15["id"] > 40000) & (id_and_house_15["id"] < 50000)]
pa_pop_15 = pa_households_15["households"].sum()
pa_pop_15


# In[21]:


# population for virginia 2015 is 3,062,783
va_households_15 = id_and_house_15.loc[(id_and_house_15["id"] > 50000) & (id_and_house_15["id"] < 54000)]
va_pop_15 = va_households_15["households"].sum()
va_pop_15


# In[22]:


# population for west virginia 2015 is 740,890
wv_households_15 = id_and_house_15.loc[(id_and_house_15["id"] > 54000)]
wv_pop_15 = wv_households_15["households"].sum()
wv_pop_15
pop_2015 = dict()
pop_2015.update({"Kentucky": 1708499, "Ohio": 4585084, "Pennsylvania": 4958859, "Virginia": 3062783, "West Virginia": 740890})
pop_2015


# In[23]:


# kentucky 2016: 1,718,217
ky_households_16 = id_and_house_16.loc[id_and_house_16["id"] < 30000]
ky_pop_16 = ky_households_16["households"].sum()
ky_pop_16


# In[24]:


# ohio 2016: 4,601,449
oh_households_16 = id_and_house_16.loc[(id_and_house_16["id"] > 30000) & (id_and_house_16["id"] < 40000)]
oh_pop_16 = oh_households_16["households"].sum()
oh_pop_16


# In[25]:


# population for pennsylvania 2016 is 4,961,929
pa_households_16 = id_and_house_16.loc[(id_and_house_16["id"] > 40000) & (id_and_house_16["id"] < 50000)]
pa_pop_16 = pa_households_16["households"].sum()
pa_pop_16


# In[26]:


# population for virginia 2016 is 3,090,178
va_households_16 = id_and_house_16.loc[(id_and_house_16["id"] > 50000) & (id_and_house_16["id"] < 54000)]
va_pop_16 = va_households_16["households"].sum()
va_pop_16


# In[184]:


# population for west virginia 2016 is 739,397
wv_households_16 = id_and_house_16.loc[(id_and_house_16["id"] > 54000)]
wv_pop_16 = wv_households_16["households"].sum()
wv_pop_16
pop_2016 = dict()
pop_2016.update({"Kentucky": ky_pop_16*2.48, "Ohio": oh_pop_16*2.41, "Pennsylvania": pa_pop_16*2.42, "Virginia": va_pop_16
                 *2.60, "West Virginia": wv_pop_16*2.4})
pop_2016


# # Equation for our mathematical model
# 
# #### Assumptions
# - The population stays constant
# - Individuals who have currently stopped using drugs can no longer be a non drug-user
# - Individuals who are abstinent have a chance to go back to being a drug user
# - Each drug report represents one drug user
# - Assume S₀ is equal to the number of households minus the number of drug reports
# - Assume I₀ is equal to number of total drug reports
# - Assume R₀ is zero
# 
# #### Variables & Equation
# We modified the SIR model for spread of diseases to create the equation for our mathematical model.
# 
# S = non drug-users
# I = drug users
# R = individuals who have currently stopped using drugs (sober) 
# N = total popuation
# r = rate of change from non-users to users 
# v = rate at which drug-users become abstinent (assume equal 0.47)
# p = rate at which individuals who are abstinent start using drugs again (assume equal to 0.85)(https://www.ashleytreatment.org/drug-addiction-recovery-statistics/)
# t = time
# 
# Our mathematical model consists of three equations:
# dS/dt = -rS(t) * I(t)  
# dI/dt = rS(t) * I(t) - vI(t) + pR(t)  
# dR/dt = vI(t) - pR(t)   

# In[178]:


# the mathematical model 
def my_model(state, t):
    S, I, R = state
    S_prime = (-0.04031)/(S+I+R) * S * I
    I_prime = (0.04031/(S+I+R)) * S * I - 0.47 * I + 0.85 * R
    R_prime = 0.47 * I - 0.85 * R
    return(S_prime, I_prime, R_prime)


# In[179]:


# define variables
t_interval = np.linspace(0, 15)
initial_state = (38162380, 240698, 0)
result = odeint(my_model, initial_state, t_interval)


# In[180]:


plt.plot(t_interval, result[:, 1])
plt.plot(t_interval, result[:, 2])


# In[181]:


# drug user population prediction for virginia

# 2010 virginia # of households * average household size - # of drug users
va2010_s = 4552270 * 2.75 - 6218
va2010_state = (va2010_s, 6218, 0) # create population vector for virginia 2010

# plug poplulation into model
time = np.linspace(0,10)
va2010_graph = odeint(my_model, va2010_state, time)
plt.plot(time, va2010_graph[:,1])
plt.plot(time, va2010_graph[:,2])
plt.title("Population of Drug Users and Abstinent users in 2010 VA")


# In[182]:


# top 5 counties in virginia for drug reports : 'FAIRFAX', 'RICHMOND', 'PRINCE WILLIAM', 'CHESTERFIELD', 'HENRICO'
data2010.loc[data2010["GEO.display-label"].str.contains("Fairfax")]
fairfax_2010 = 381768 * 2.75 - 117 # non drug-user pop for fairfax county
fairfax2010_state = (fairfax_2010, 117, 0)

data2010.loc[data2010["GEO.display-label"].str.contains("Richmond")]
richmond_2010 = 3007 * 2.75 - 66 # non drug-user pop for richmond county
richmond2010_state = (richmond_2010, 66, 0)

data2010.loc[data2010["GEO.display-label"].str.contains("Prince William")]
prince_2010 = 124879 * 2.75 - 103 # non drug-user pop for Prince William county
prince2010_state = (prince_2010, 103, 0)

data2010.loc[data2010["GEO.display-label"].str.contains("Chesterfield")]
chesterfield_2010 = 112404 * 2.75 - 105 # non-druguser pop for Chesterfield county
chesterfield2010_state = (chesterfield_2010, 105, 0)

data2010.loc[data2010["GEO.display-label"].str.contains("Henrico")]
henrico_2010 = 121767 * 2.75 - 101 # non-druguser pop for Henrico county
henrico2010_state = (henrico_2010, 101, 0)


# In[333]:


# graph for Fairfax County, VA
fairfax_graph = odeint(my_model, fairfax2010_state, time)
plt.plot(time, fairfax_graph[:,1])
plt.plot(time, fairfax_graph[:,2])
plt.title("Number of Drug Users and Number of Recovered Over Time (Fairfax, VA)")
plt.xlabel("Years from 2010")
plt.ylabel("Number of people")
plt.savefig("fairfax.jpg", bbox_inches="tight")


# In[150]:


richmond_graph = odeint(my_model, richmond2010_state, time)
plt.plot(time, richmond_graph[:,1])
plt.plot(time, richmond_graph[:,2])


# In[175]:


prince_graph = odeint(my_model, prince2010_state, time)
plt.plot(time, prince_graph[:,1])
plt.plot(time, prince_graph[:,2])


# In[151]:


chesterfield_graph = odeint(my_model, chesterfield2010_state, time)
plt.plot(time, chesterfield_graph[:,1])
plt.plot(time, chesterfield_graph[:,2])


# In[152]:


henrico_graph = odeint(my_model, henrico2010_state, time)
plt.plot(time, henrico_graph[:,1])
plt.plot(time, henrico_graph[:,2])


# In[115]:


# drug user population prediction for west virginia
# 2010 west virginia # of households * average household size - # of drug users
wv2010_s = 740874 * 2.43 - 1699
wv2010_state = (wv2010_s, 1699, 0)

# plug population into model
wv2010_graph = odeint(my_model, wv2010_state, time)
plt.plot(time, wv2010_graph[:,1])
plt.plot(time, wv2010_graph[:,2])


# In[166]:


# top 5 counties in West Virginia for drug reports: 'KANAWHA', 'BERKELEY', 'CABELL', 'RALEIGH', 'HARRISON'
data2010.loc[data2010["GEO.display-label"].str.contains("Kanawha")]
kanawha_2010 = 82501 * 2.43 - 84
kanawha_state = (kanawha_2010, 84, 0)

data2010.loc[data2010["GEO.display-label"].str.contains("Berkeley")]
berkeley_2010 = 38730 * 2.43 - 65
berkeley_state = (berkeley_2010, 65, 0)

data2010.loc[data2010["GEO.display-label"].str.contains("Cabell")]
cabell_2010 = 40564 * 2.43 - 55
cabell_state = (cabell_2010, 55, 0)

data2010.loc[data2010["GEO.display-label"].str.contains("Raleigh")]
raleigh_2010 = 31196 * 2.43 - 68
raleigh_state = (raleigh_2010, 68, 0)

data2010.loc[data2010["GEO.display-label"].str.contains("Harrison")]
harrison_2010 = 27740 * 2.43 - 56
harrison_state = (harrison_2010, 56, 0)


# In[334]:


# Kanawha county, WV
kanawha_graph = odeint(my_model, kanawha_state, time)
plt.plot(time, kanawha_graph[:,1])
plt.plot(time, kanawha_graph[:,2])
plt.title("Number of Drug Users and Number of Recovered Over Time (Kanawha, WV)")
plt.xlabel("Years from 2010")
plt.ylabel("Number of people")
plt.savefig("kanawha.jpg", bbox_inches="tight")


# In[162]:


berkeley_graph = odeint(my_model, berkeley_state, time)
plt.plot(time, berkeley_graph[:,1])
plt.plot(time, berkeley_graph[:,2])


# In[163]:


cabell_graph = odeint(my_model, cabell_state, time)
plt.plot(time, cabell_graph[:,1])
plt.plot(time, cabell_graph[:,2])


# In[164]:


raleigh_graph = odeint(my_model, raleigh_state, time)
plt.plot(time, raleigh_graph[:,1])
plt.plot(time, raleigh_graph[:,2])


# In[167]:


harrison_graph = odeint(my_model, harrison_state, time)
plt.plot(time, harrison_graph[:,1])
plt.plot(time, harrison_graph[:,2])


# In[312]:


a, intercept = np.polyfit(t_interval[6:51], fairfax_graph[6:, 1], 1)
b, intercept = np.polyfit(t_interval[6:51], richmond_graph[6:, 1], 1)
c, intercept = np.polyfit(t_interval[6:51], prince_graph[6:, 1], 1)
d, intercept = np.polyfit(t_interval[6:51], chesterfield_graph[6:, 1], 1)
e, intercept = np.polyfit(t_interval[6:51], henrico_graph[6:, 1], 1)
f, intercept = np.polyfit(t_interval[6:51], kanawha_graph[6:, 1], 1)
g, intercept = np.polyfit(t_interval[6:51], berkeley_graph[6:, 1], 1)
h, intercept = np.polyfit(t_interval[6:51], cabell_graph[6:, 1], 1)
i, intercept = np.polyfit(t_interval[6:51], raleigh_graph[6:, 1], 1)
j, intercept = np.polyfit(t_interval[6:51], harrison_graph[6:, 1], 1)
slopes = [a, b, c, d, e, f, g, h, i, j]
slopes_df = pd.DataFrame(data = {"Number of people becoming drug users per year":slopes},
                         index = ["va1", "va2", "va3", "va4", "va5", "wv1", "wv2", "wv3", "wv4", "wv5"])
slopes_df


# The county with the greatest slope in Virginia is Fairfax, with 1.27. 
# The county with the greatest slope in West Virginia is Kanawha with 0.912.
# Therefore, the predicted counties where opioid abuse might have started for state is Fairfax and Kanawha. 

# Drug Reports in 2010
# 
# Kentucky: 624
# Ohio: 686
# Pennsylvania: 501
# Virginia: 822
# West Virginia: 203
# 
# Percentage of total population:
# Kentucky 

# In[300]:


data2010["GEO.id2"] = pd.to_numeric(data2010["GEO.id2"][1:,])
ky2010data = data2010.loc[data2010["GEO.id2"] < 30000]
oh2010data = data2010.loc[(data2010["GEO.id2"] > 30000) & (data2010["GEO.id2"] < 40000)]
pa2010data = data2010.loc[(data2010["GEO.id2"] > 40000) & (data2010["GEO.id2"] < 50000)]
va2010data = data2010.loc[(data2010["GEO.id2"] > 50000) & (data2010["GEO.id2"] < 54000)]
wv2010data = data2010.loc[(data2010["GEO.id2"] > 54000)]


# In[299]:


data2016["GEO.id2"] = pd.to_numeric(data2016["GEO.id2"][1:,])
ky2016data = data2016.loc[data2016["GEO.id2"] < 30000]
oh2016data = data2016.loc[(data2016["GEO.id2"] > 30000) & (data2016["GEO.id2"] < 40000)]
pa2016data = data2016.loc[(data2016["GEO.id2"] > 40000) & (data2016["GEO.id2"] < 50000)]
va2016data = data2016.loc[(data2016["GEO.id2"] > 50000) & (data2016["GEO.id2"] < 54000)]
wv2016data = data2016.loc[(data2016["GEO.id2"] > 54000)]


# In[301]:


# change first row to header
headers = data2016.iloc[0].values
data2016.columns = headers
data2016.drop(index=0, axis=0, inplace=True)


# In[313]:


data2016["Percent; VETERAN STATUS - Civilian population 18 years and over"]


# In[314]:


# "HC03_VC99" - veteran, "HC03_VC87" - diploma, "HC03_VC14" - living alone
ky2016_veteran = pd.to_numeric(ky2016data["Percent; VETERAN STATUS - Civilian population 18 years and over"], errors='coerce').mean()
ky2016_diploma = pd.to_numeric(ky2016data["Percent; EDUCATIONAL ATTAINMENT - High school graduate (includes equivalency)"], errors='coerce').mean()
ky2016_single = pd.to_numeric(ky2016data["Percent; HOUSEHOLDS BY TYPE - Nonfamily households - Householder living alone"], errors='coerce').mean()
ky2016_rates = (ky2016_veteran,ky2016_diploma,ky2016_single)


# In[272]:


data2010


# In[ ]:




