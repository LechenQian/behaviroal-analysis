#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:32:11 2019

@author: lechenqian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import random
import matplotlib as mpl
import datetime 


#%%  Functions
def read_data(filedir):

    filename = []
    for dirpath, dirnames, files in os.walk(filedir):
    #     print(f'Found directory: {dirpath}')
        for f_name in files:
            if f_name.endswith('.xlsx'):
                filename.append(dirpath+'/'+f_name)
        
    print(filename)
    return filename


#Dict for data for each days
def create_dict_mouse_day_df(df,filename):
    date_key_list = []
    for file in filename:
        date = file[62:72] #number for date
        
        date_key_list.append(date)
        data = pd.read_excel(file)
        data.columns = ['Time','Event']
        df.update({date:data})
    return df,date_key_list

def sort_date(date_key_list):
    from datetime import datetime
    #date_key_list.sort(key = lambda date: datetime.strptime(date, '%Y-%M-%d'))
    dates = [datetime.strptime(ts, "%Y-%m-%d") for ts in date_key_list]
    dates.sort()
    sorteddates = [datetime.strftime(ts, "%Y-%m-%d") for ts in dates]   
    print(sorteddates)
    return sorteddates


def delete_date(date_list,dates):
    for date in dates:
        date_list.remove(date)
    print(date_list)
    return date_list

def convert_event_to_trial_df(df):
    for key, value in df.items():
        try:
            new_df = generate_trials_dataframe(value)
            df[key] = new_df
        except:
            pass
    return df

def generate_trials_dataframe(ori_df):
    
    trials, go_odor_on, go_odor_off, nogo_odor_on, nogo_odor_off,water_on, water_off, trial_end = seperate_events(ori_df)
    d = {'go_odor_on': go_odor_on, 'go_odor_off': go_odor_off,'nogo_odor_on': nogo_odor_on, 
         'nogo_odor_off': nogo_odor_off, 'water_on':water_on,'water_off':water_off,'licking':trials,
         'trial_end':trial_end}
    df = pd.DataFrame(data = d)
    return df
    
def seperate_events(df):
    
    start_trials = 0
    trials        = []
    go_odor_on    = []
    go_odor_off   = []
    nogo_odor_on  = []
    nogo_odor_off = []
    water_on      = []
    water_off     = []
    trial_end     = []
    
    for index, row in df.iterrows():
        if row['Event'] == 101:

            start_trials = row['Time']
            
            temp_licks = []
            temp_go_odor_on = []
            temp_go_odor_off = []
            temp_nogo_odor_on = []
            temp_nogo_odor_off = []
            temp_water_on = []
            temp_water_off = []
            temp_trial_end = []
        elif row['Event'] == 11:
            lick_time = row['Time'] - start_trials
            temp_licks.append(lick_time)
        elif row['Event'] == 131:
            temp_go_odor_on.append(row['Time']- start_trials)


        elif row['Event'] == 130:
            temp_go_odor_off.append(row['Time']- start_trials)
        elif row['Event'] == 141:
            temp_nogo_odor_on.append(row['Time']- start_trials)


        elif row['Event'] == 140:
            temp_nogo_odor_off.append(row['Time']- start_trials)
        elif row['Event'] == 51:
            temp_water_on.append(row['Time']- start_trials)


        elif row['Event'] == 50:
            temp_water_off.append(row['Time']- start_trials)
        elif row['Event'] == 100:
            temp_trial_end.append(row['Time']- start_trials)

            trials.append(temp_licks)
            go_odor_on.append(temp_go_odor_on)
            go_odor_off.append(temp_go_odor_off)
            nogo_odor_on.append(temp_nogo_odor_on)
            nogo_odor_off.append(temp_nogo_odor_off)
            water_on.append(temp_water_on)
            water_off.append(temp_water_off)
            trial_end.append(temp_trial_end)

    return  trials, go_odor_on, go_odor_off, nogo_odor_on, nogo_odor_off,water_on, water_off, trial_end

def generate_islicking_isnolicking(df):
  
    islicking = []
    isnolicking = []
    for index, row in df.iterrows():
        
        # cont go reward
        if len(row['go_odor_on']) !=  0 :
            if any(x > row['go_odor_off'][0] and x < row['go_odor_off'][0]+2.5 for x in row['licking']):
                islicking.append(1)
                isnolicking.append(np.NaN)
            else:
                islicking.append(0)
                isnolicking.append(np.NaN)
                
        elif len(row['nogo_odor_on']) !=  0 :
            if any(x > row['nogo_odor_off'][0] and x < row['nogo_odor_off'][0]+2.5 for x in row['licking']):
                islicking.append(np.NaN)
                isnolicking.append(0)
            else:
                islicking.append(np.NaN)
                isnolicking.append(1)
        else:
            islicking.append(np.NaN)
            isnolicking.append(np.NaN)
            
    return islicking, isnolicking
def add_islicking_for_selected(df,days):
    for day in days:
        islicking, isnolicking = generate_islicking_isnolicking(df[day])
        df[day]['licking for go'] = islicking
        df[day]['nolicking for nogo'] = isnolicking
    return df

def cal_hit_correj_rate_every_n(df,every_n_trial = 20):
    p_hit_every_n = []
    p_correj_every_n = []
    count_hit = 0
    count_goodor = 0
    count_nogoodor = 0
    count_correj = 0
    for index, row in df.iterrows():


        if len(row['go_odor_on']) !=  0:
            count_goodor+=1
            if any(x > row['go_odor_off'][0] and x < row['go_odor_off'][0]+2.5 for x in row['licking']):
                count_hit +=1
        if len(row['nogo_odor_on']) !=  0:
            count_nogoodor+=1
            if any(x > row['nogo_odor_on'][0] and x < row['nogo_odor_off'][0]+2.5 for x in row['licking']):
                count_correj -= 1
        if index%every_n_trial == every_n_trial-1:
            try:
                prob_hit = count_hit/count_goodor
            except:
                prob_hit = np.nan
            try:       
                prob_correj = (count_nogoodor+count_correj)/count_nogoodor
            except:
                prob_correj = np.nan

            p_hit_every_n.append(prob_hit)
            p_correj_every_n.append(prob_correj)      
            count_hit = 0
            count_goodor = 0
            count_nogoodor = 0
            count_correj = 0
  

         
    return p_hit_every_n, p_correj_every_n


# Average licking number and latency for every 20 for odor on trials
def ave_licking_number_latency_every_n(df,every_n_trial):

    
    licking_every_n = []
    latency_every_n = []
    baselicking_every_n = []
    
    count_licking_base             = 0
    count_time_no_baseline         = 0
    count_licking_base_no_baseline = 0
    count_time                     = 0 #baseline time
    count_licking                  = 0
    count_latency                  = 0
    count_odoron = 0
    is_no_baseline = 0
    for index, row in df.iterrows():


        if len(row['go_odor_on']) !=  0:
            
            # licking and latencu after go odor presentation within action window
            count_odoron += 1
            x = [i for i in row['licking']  if i> row['go_odor_off'][0] and i< row['go_odor_off'][0]+2.5]
            count_licking += len(x)
            if len(x) != 0:
                count_latency += min(x)-row['go_odor_off'][0]
            else:
                count_latency += 2.5
                
            # prepare for no_baseline
            count_time_no_baseline += 2.5
            x = [i for i in row['licking']  if i> 0 and i< 2.5]
            count_licking_base_no_baseline += len(x)
        elif len(row['go_odor_on']) ==0 and len(row['nogo_odor_on']) ==  0:
            count_time += 1+2.5

            x = [i for i in row['licking']  if i> 2.5 and i< 6.0]
            count_licking_base += len(x)


        if index%every_n_trial == every_n_trial-1:
            licking_every_n.append(count_licking/count_odoron)
            latency_every_n.append(count_latency/count_odoron)
            try:
                baselicking_every_n.append(count_licking_base/count_time)
            except:
                baselicking_every_n.append(count_licking_base_no_baseline/count_time_no_baseline)
                
                

            count_licking_base = 0
            count_time = 0
            count_licking = 0
            count_odoron = 0
            count_latency = 0
        
    
    return licking_every_n,latency_every_n,baselicking_every_n

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def extract_specified_odor_trials(df, odortype = 'go'):
    rowname_on = odortype + '_odor_on'
    rowname_off = odortype + '_odor_off'
    specified_odor_licking_list = []
    for index, row in df.iterrows():
        if len(row[rowname_on]) !=  0:
            specified_odor_licking_list.append(row['licking'])         
    return specified_odor_licking_list

def bin_licking(nested_list, binnum, end_t = 10, start_t = 0):
    bins = np.linspace(start_t, end_t, binnum)
    binned_licking_list = []
    for data in nested_list:
        binned = np.histogram(data, bins)
        binned_licking_list.append(binned[0])
        binned_time_list = binned[1]
    return binned_licking_list, binned_time_list

        

#%%
 ### 

#%%
mouse_id = 'C20'
filedir= '/Users/lechenqian/OneDrive - Harvard University/2019-data/{}'.format(mouse_id)
filename = read_data(filedir)
# 
df={}
df,date_key_list = create_dict_mouse_day_df(df,filename)
date_key_list = sort_date(date_key_list)  

#%%
del_dates = [] ###
date_list = delete_date(date_key_list,del_dates)

#%%
training_days = date_list[0:7]
degradation_days = date_list[7:]
df_trials_C17 = convert_event_to_trial_df(df)
 
df_trials_C17 = add_islicking_for_selected(df,training_days)

#%%
df_date = []
df_hit = []
df_correj = []

licking_aft_odoron = []
latency_aft_odoron = []
df_baselick = []
for date in date_list: # train_day_list
    
    p_hit_every20,p_correj_every20 = cal_hit_correj_rate_every_n(df_trials_C17[date],160)
    licking_every20,latency_every20, baselicking_every20 = ave_licking_number_latency_every_n(df_trials_C17[date],160)
    
    df_hit.append(p_hit_every20)
    df_correj.append(p_correj_every20)
    licking_aft_odoron.append(licking_every20)
    latency_aft_odoron.append(latency_every20)
    df_baselick.append(baselicking_every20)
    print(date,'   All set!')
    
# print(df_hit)
# print(df_correj)
# print(licking_aft_odoron)
# print(latency_aft_odoron)
# print(df_baselick)

data = {'df_hit':df_hit,'df_correj':df_correj,'licking_aft_odoron':licking_aft_odoron,'latency_aft_odoron':latency_aft_odoron,'baselicking':df_baselick}
Statistics_C17 = pd.DataFrame(data)  

#%%
fig,ax = plt.subplots(nrows=5, ncols=1, sharex=True,figsize = (10,27))
index1 = 0
c1='#F30021' #red
c2='#FFA100' #yellow
c3 = '#5033DD'
c4 = '#2BF39E'
c5 = '#5E23A0'
c6 = '#F4758A'
c7 = '#271719'
c8 = '#BFAEB0'
font = {'size': 7}

mpl.rc('font', **font)
for index, row in Statistics_C17.iterrows():
    index2 = index1 + len(row['licking_aft_odoron'])
    x  = range(index1,index2)
    
    ax[0].plot(x, row['df_hit'],'^-',linewidth = 3,alpha = 0.8,color=colorFader(c1,c2,(index+1)/len(date_list)), label = 'hit' if index == 1 else '')
    ax[1].plot(x, row['df_correj'],'^-',linewidth = 3,alpha = 0.5,color=colorFader(c3,c4,(index+1)/len(date_list)),label = 'correct rejection' if index == 1 else '')
    ax[0].legend(loc='lower right',frameon = False)
    ax[1].legend(loc='lower right',frameon = False)
    ax[0].set_ylabel('percentage(%)')
    ax[1].set_ylabel('percentage(%)')
    
    ax[2].plot(x, row['licking_aft_odoron'],'^-',linewidth = 3,alpha = 0.8,color=colorFader(c5,c6,(index+1)/len(date_list)))
    ax[2].set_ylabel('# licking \n in action window')
    ax[3].plot(x, row['latency_aft_odoron'],'^-',linewidth = 3,alpha = 0.8,color=colorFader(c5,c6,(index+1)/len(date_list)))
    ax[3].set_ylabel('latency to lick\n in action window')
    ax[4].plot(x, row['baselicking'],'^-',linewidth = 3,alpha = 0.8,color=colorFader(c7,c8,(index+1)/len(date_list)))
    ax[4].set_ylabel('# baseline licking')
   
   
    
    index1 = index2
ax[0].spines["top"].set_visible(False)
ax[1].spines["top"].set_visible(False)
ax[2].spines["top"].set_visible(False)
ax[3].spines["top"].set_visible(False)
ax[4].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)    
ax[1].spines["right"].set_visible(False)    
ax[2].spines["right"].set_visible(False)    
ax[3].spines["right"].set_visible(False)    
ax[4].spines["right"].set_visible(False) 
plt.xticks([])
save_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
plt.savefig("/Users/lechenqian/OneDrive - Harvard University/2019-data/{}/{}.png".format(mouse_id,save_time), bbox_inches="tight")

plt.show() 

#%%
# PCA trajectory of latency after odor on
latency_matrix = np.zeros([8,len(latency_aft_odoron)])
for index, item in enumerate(latency_aft_odoron):
    temp = np.asarray(item[0:8]).T
    latency_matrix[:, index] = temp
    
latency_cov = np.cov(latency_matrix)
eigenvalue,eigenvec = np.linalg.eigh(latency_cov) # eigendecomposition
sort_index = np.argsort(-eigenvalue)
eigenvec_sorted = eigenvec[:,sort_index]
eigenvalue_sorted = eigenvalue[sort_index]
first_eigen = eigenvec_sorted[:,0]
second_eigen = eigenvec_sorted[:,1]
latency_first_proj = first_eigen.T.dot(latency_matrix)
latency_second_proj = second_eigen.T.dot(latency_matrix)

    
    
#%%
# Topics: line, color, LineCollection, cmap, colorline, codex
'''
Defines a function colorline that draws a (multi-)colored 2D line with coordinates x and y.
The color is taken from optional data in z, and creates a LineCollection.

z can be:
- empty, in which case a default coloring will be used based on the position along the input arrays
- a single number, for a uniform color [this can also be accomplished with the usual plt.plot]
- an array of the length of at least the same length as x, to color according to this data
- an array of a smaller length, in which case the colors are repeated along the curve

The function colorline returns the LineCollection created, which can be modified afterwards.

See also: plt.streamplot
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# Data manipulation:

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments


# Interface to LineCollection:

def colorline(x, y, z=None, cmap=plt.get_cmap('cool'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap,  linewidth=linewidth, alpha=alpha)
    
    ax = plt.gca()
    ax.add_collection(lc)
    
    return lc
        
    
def clear_frame(ax=None): 
    # Taken from a post by Tony S Yu
    if ax is None: 
        ax = plt.gca() 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 
    for spine in ax.spines.itervalues(): 
        spine.set_visible(False) 
#%%
fig, axes = plt.subplots()

colorline(-latency_first_proj,latency_second_proj,cmap="cool")
plt.scatter(-latency_first_proj[0],latency_second_proj[0],color = 'red',s = 1)
plt.xlim(-1,5)
plt.ylim(-1, 10)


plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#%%
### Event Plot
def event_plot(df,exp_date,gocolor = 'purple',nogocolor = 'yellow', rewardcolor = 'blue',lickcolor = 'grey'):
    import datetime
    lineoffsets2 = 1
    linelengths2 = 1
    df = df[exp_date]

    # create a horizontal plot
    plt.figure(figsize=(10,8))

    for i in range(len(df.go_odor_on)):
        plt.hlines(i, df.go_odor_on[i], df.go_odor_off[i],color = gocolor,alpha = 1,
                   linewidth = 1,label = 'go odor' if i ==0 else '')
        plt.hlines(i, df.water_on[i], df.water_off[i],color = rewardcolor,
                   linewidth = 1,label = 'water' if i ==0 else '')
        plt.hlines(i, df.nogo_odor_on[i], df.nogo_odor_off[i],color = nogocolor,alpha = 1,
                   linewidth = 1,label = 'no go odor' if i ==0 else '')
    plt.eventplot(df.licking, colors=lickcolor, lineoffsets=lineoffsets2,
                        linelengths=linelengths2,alpha = 0.5, label = 'licking')
    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
   
    ax.spines["right"].set_visible(False)    
    draw_loc = ax.get_xlim()[1]
    for i in range(len(df.go_odor_on)):
        if ~ np.isnan(df['licking for go'][i]):
            if df['licking for go'][i]:
                plt.hlines(i, draw_loc+1, draw_loc+1.5,color = 'green',alpha = 1,
                   linewidth = 1 )
            else:
                plt.hlines(i, draw_loc+1, draw_loc+1.5,color = 'red',alpha = 1,
                   linewidth = 1 )
        elif ~ np.isnan(df['nolicking for nogo'][i]):
            if df['nolicking for nogo'][i]:
                plt.hlines(i, draw_loc+1.8, draw_loc+2.3,color = 'green',alpha = 1,
                   linewidth = 1 )
            else:
                plt.hlines(i, draw_loc+1.8, draw_loc+2.3,color = 'red',alpha = 1,
                   linewidth = 1 )
            
                
                    
        

    #plt.yticks(range(0, 91, 10), [str(x) + "%" for x in range(0, 91, 10)], fontsize=14)    
    plt.xticks(fontsize=14)    
    ax.set_xlim([ax.get_xlim()[0],draw_loc+4])
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")    

    plt.ylabel('Trials',fontsize = 16)
    plt.xlabel('Time',fontsize = 16)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),prop={'size': 12},loc='upper center', bbox_to_anchor=(0.5, 1.05),
          frameon=False,fancybox=False, shadow=False, ncol=4)
    datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    #plt.savefig("/Users/lechenqian/OneDrive - Harvard University/2019-data/{}/{}/{}.png".format(mouse_id,exp_date,datetime.date.today()), bbox_inches="tight"); 
    plt.show()
#%%
event_plot(df_trials_C17,training_days[0])
    
#%% licking number after go, no go, and empty trial
def licking_for_trialtype_training(df):
    
    count_licking_go      = 0
    count_licking_nogo    = 0
    count_licking_empty   = 0
    count_go_trial        = 0 #baseline time
    count_nogo_trial      = 0
    count_empty_trial     = 0
    
    for index, row in df.iterrows():

        # go trial
        if len(row['go_odor_on']) !=  0:
            
            # licking after go odor presentation within action window
            count_go_trial += 1
            x = [i for i in row['licking']  if i> row['go_odor_off'][0] and i< row['go_odor_off'][0]+2.5]
            count_licking_go += len(x)/2.5
            
            # baseline licking after go
            #count_empty_trial += 1

            #x_pre = [i for i in row['licking']  if i> 0 and i< row['go_odor_on'][0] ]
            #x_post = [i for i in row['licking']  if i> row['go_odor_off'][0]+2.5+4 ]
            
            #count_licking_empty += len(x_pre)+len(x_post)/3.5 ## I DON'T the ending time!! reprocess the data
            
            
        
        
        
        # no go trial
        if len(row['nogo_odor_on']) !=  0:
            
            # licking after go odor presentation within action window
            count_nogo_trial += 1
            x = [i for i in row['licking']  if i> row['nogo_odor_off'][0] and i< row['nogo_odor_off'][0]+2.5]
            count_licking_nogo += len(x)/2.5
        
     
        # empty trial
        elif len(row['go_odor_on']) ==0 and len(row['nogo_odor_on']) ==  0:
            count_empty_trial += 1

            x = [i for i in row['licking']  if i> 2.5 and i< 6.0]
            count_licking_empty += len(x)/3.5


 
        try:
            ave_licking_go = count_licking_go/count_go_trial
        except:
            ave_licking_go = np.NaN
            
        try:
            ave_licking_nogo = count_licking_nogo/count_nogo_trial
        except:
            ave_licking_nogo = np.NaN
        try:
            ave_licking_empty = count_licking_empty/count_empty_trial
        except:
            ave_licking_empty = np.NaN
    
    
    return ave_licking_go, ave_licking_nogo, ave_licking_empty

#%%

ave_licking_go_list = []
ave_licking_nogo_list = []
ave_licking_empty_list = []
   
for day in training_days:
    temp_go,temp_nogo, temp_empty = licking_for_trialtype_training(df[day]) 
    ave_licking_go_list.append(temp_go)
    ave_licking_nogo_list.append(temp_nogo)
    ave_licking_empty_list.append(temp_empty)

plt.figure(figsize = (10,5))  
xrange = np.array(range(len(training_days)))
plt.plot(xrange,ave_licking_go_list,'^-')
plt.plot(xrange,ave_licking_nogo_list,'^-')
plt.plot(xrange,ave_licking_empty_list,'^-')
plt.show()
 
#%%

    
   






























