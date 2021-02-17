#!/usr/bin/env python
# coding: utf-8

# # Importing The Libraries
# ###### Import the essential libraries to get started!

# In[91]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Fetch the Data
# ###### Since, I have already cleaned the Titanic data-set in my previous notebook. Hence, I will be using the cleaned Data-set

# In[92]:


df = pd.read_csv('Titanic_Na_less.csv')


# In[95]:


df.head()


# ## Analysis over the "Survived" column

# In[96]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df)


# ###### Unfortunately, more people died than survived.

# In[97]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df)


# ###### More Male passengers died, as compared to Female passengers. But will dig deeper to find the logic behind this result

# ## Let's check the total percentage of Male and Female in general

# In[98]:


df['Sex'].value_counts(normalize=True)


# In[99]:


sns.set_style('whitegrid')
sns.countplot(x='Sex',data=df)


# #### Seems like, there were already more Male passengers as compared to female in general. Therefore, that can be a reason behind the more death of the Male passengers.

# ## Feature Engineering
# #### Let's categorize the Age column

# In[100]:


def process_age(df,cut_points,label_names):
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [0,5,12,18,35,60,100]
label_names = ["Infant","Child","Teenager","Young Adult","Adult","Senior"]
df = process_age(train,cut_points,label_names)


# In[101]:


df["Age_categories"].value_counts()


# In[102]:


sns.countplot(x="Age_categories", data=df)


# ### Young Adults had the majority in the list of passengers

# ##### Let's check their survival rate

# In[103]:


sns.set_style('whitegrid')
sns.countplot(x='Age_categories',hue='Survived',data=df,palette='rainbow')


# ###### It clearly states that, Young Adults had less survival rate, amongst all. However, Infant's rate of survival is the highest. 
# *Seems like a great sacrifice by the Young Adults in order to save the fellow passengers.*

# #### As I observed a pattern in the name, let's see if the Title(the prefix of name) has to do anything with the survival.

# In[134]:


def get_title(x):
    return x.split(',')[1].split('.')[0].strip()


# In[135]:


df['Title'] = df['Name'].apply(get_title)


# In[165]:


Tot_Pass = df["Title"].value_counts()
Tot_Pass

#Total Number of passengers according to their titles


# In[166]:


Surv_Pass = df.groupby("Title").sum()["Survived"]
Surv_Pass
#Number of passengers who survived


# In[196]:


Dead_Pass = df["Title"].value_counts() - df.groupby("Title").sum()["Survived"]
Dead_Pass
#If we minus from the number of Total passengers by the Total number of surviving passengers.
#Then we can get this observation of number of death as per the title


# In[199]:


title_df = pd.concat([Tot_Pass, Dead_Pass, Surv_Pass], axis=1)
title_df


# In[ ]:





# In[201]:


title_df.columns = ["TotalNo. of Pass", "Dead", "Survived"]


# In[202]:


title_df


# In[210]:


Perc_Pass = Surv_Pass / Tot_Pass * 100
Perc_Pass


# In[212]:


title_df = pd.concat([Tot_Pass, Dead_Pass, Surv_Pass], axis=1)


# In[241]:


cm = sns.light_palette("green", as_cmap=True)
# using seaborn color palette as well as 
# in each column 
 
title_df.style.background_gradient(cmap=cm).set_precision(1).highlight_min(axis=1,color='red').highlight_max(axis=0, color='blue')


# ###### As per the observance: Reverend, Don, Jonkheer & Captain couldn't survive.
# ###### Mr. also had the low rate in terms of Survival, however this title in particular was in the majority in the list of passengers.
# ###### Those who had the highest survival rate, they belonged to a higher class in the society.
# ###### Age played the major role in the survival, thus the younger the passenger the higher the chance of survival
# 

# # Let's observe some more patterns!

# In[206]:


sns.pairplot(data=df, hue="Survived")


# ## To be continued!!

# In[ ]:




