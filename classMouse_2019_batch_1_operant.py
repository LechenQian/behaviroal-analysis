#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:40:48 2019

@author: lechenqian
"""
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import random
import matplotlib as mpl


class Mouse:
    def __init__(self,mouse_id):
        self.mouse_id = mouse_id
        self.filedir = '/Users/lechenqian/OneDrive - Harvard University/2019-data/{}'.format(self.mouse_id)
        self.filename = ''
        self.selected_filename = ''
        self.all_days = []
        self.sorted_all_days = []
        self.training_days = []
        self.degradation_days = []
        self.df_trials = {}
        self.df_eventcode = {}
        self.p_hit = {}
        self.p_correj = {}
        self.licking_actionwindow = {}
        self.licking_latency = {}
        self.licking_baselicking = {}
        self.stats = {}
        self.event_data = ''
        


        
        
    def read_data(self):
        filedir = self.filedir
        filename = []
        for dirpath, dirnames, files in os.walk(filedir):
        #     print(f'Found directory: {dirpath}')
            for f_name in files:
                if f_name.endswith('.xlsx'):
                    filename.append(dirpath+'/'+f_name)
            
        #print(filename)
        self.filename = filename
        
    
    def select_dates(self):
        pass
        
    
    
    
    #Dict for data for each days   
    def create_dict_mouse_day_df(self, original = True):
        date_key_list = []
        df = {}
        if original ==  True:
            filenames = self.filename
        else:
            filenames = self.selected_filename
            
        for file in filenames:
            date = file[62:72] #number for date
            
            date_key_list.append(date)
            data = pd.read_excel(file)
            data.columns = ['Time','Event']
            df.update({date:data})
        self.df_eventcode = df
        self.all_days = date_key_list
        
    
    
    def sort_date(self):
        
        #date_key_list.sort(key = lambda date: datetime.strptime(date, '%Y-%M-%d'))
        dates = [datetime.strptime(ts, "%Y-%m-%d") for ts in self.all_days]
        dates.sort()
        self.sorted_all_days = [datetime.strftime(ts, "%Y-%m-%d") for ts in dates]   
    
        print(self.sorted_all_days)
        
        


    def delete_date(self, dates):
        for date in dates:
            self.sorted_all_days.remove(date)
        
        return self.sorted_all_days
        
    
    def convert_event_to_trial_df(self):
        for key, value in self.df_eventcode.items():
            print(key)
        
            new_df = self.generate_trials_dataframe(value)
            self.df_trials[key] = new_df
            print('yes!')
            
        
    
    def generate_trials_dataframe(self,ori_df):
        
        trials, go_odor_on, go_odor_off, nogo_odor_on, nogo_odor_off,water_on, water_off, trial_end = self.seperate_events(ori_df)
        d = {'go_odor_on': go_odor_on, 'go_odor_off': go_odor_off,'nogo_odor_on': nogo_odor_on, 
             'nogo_odor_off': nogo_odor_off, 'water_on':water_on,'water_off':water_off,'licking':trials,
             'trial_end':trial_end}
        df = pd.DataFrame(data = d)
        return df
    
    
    def seperate_events(self,df):
        
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
    
    
    
    
 # data analysis
 
    def add_islicking_for_selected(self,days = None):
        if days == None:
            days = self.sorted_all_days
        for day in days:
            islicking, isnolicking = self.generate_islicking_isnolicking(self.df_trials[day])
            self.df_trials[day]['licking for go'] = islicking
            self.df_trials[day]['nolicking for nogo'] = isnolicking
        
    
    def generate_islicking_isnolicking(self,value):
      
        islicking = []
        isnolicking = []
        for index, row in value.iterrows():
            
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

    def generate_Stats(self, every_n_trials = 20):    
        df_date = []
        df_hit = []
        df_correj = []
        
        licking_aft_odoron = []
        latency_aft_odoron = []
        df_baselick = []
        
        for date in self.sorted_all_days: # train_day_list
            
            p_hit_every20,p_correj_every20 = self.cal_hit_correj_rate_every_n(self.df_trials[date],every_n_trials)
            licking_every20,latency_every20, baselicking_every20 = self.ave_licking_number_latency_every_n(self.df_trials[date],every_n_trials)
            
            df_hit.append(p_hit_every20)
            df_correj.append(p_correj_every20)
            licking_aft_odoron.append(licking_every20)
            latency_aft_odoron.append(latency_every20)
            df_baselick.append(baselicking_every20)
            print(date,'   All set!')
            
        self.p_hit.update({str(every_n_trials):df_hit})
        self.p_correj.update({str(every_n_trials):df_correj})
        self.licking_actionwindow.update({str(every_n_trials):licking_aft_odoron})
        self.licking_latency.update({str(every_n_trials):latency_aft_odoron})
        self.licking_baselicking.update({str(every_n_trials):df_baselick})
        
        data = {'df_hit':df_hit,'df_correj':df_correj,'licking_aft_odoron':licking_aft_odoron,'latency_aft_odoron':latency_aft_odoron,'baselicking':df_baselick}
        Stats = pd.DataFrame(data)
        self.stats.update({str(every_n_trials):Stats})
        
            
    
    
    
    def cal_hit_correj_rate_every_n(self, df, every_n_trial = 20):
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
            
            # append statistics for every n trials into the list
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
    
    def ave_licking_number_latency_every_n(self,df,every_n_trial):
    
        
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
    
    def extract_specified_odor_trials(self,df, odortype = 'go'):
        rowname_on = odortype + '_odor_on'
        rowname_off = odortype + '_odor_off'
        specified_odor_licking_list = []
        for index, row in df.iterrows():
            if len(row[rowname_on]) !=  0:
                specified_odor_licking_list.append(row['licking'])         
        return specified_odor_licking_list
    
    def bin_licking(self,nested_list, binnum, end_t = 10, start_t = 0):
        bins = np.linspace(start_t, end_t, binnum)
        binned_licking_list = []
        for data in nested_list:
            binned = np.histogram(data, bins)
            binned_licking_list.append(binned[0])
            binned_time_list = binned[1]
        return binned_licking_list, binned_time_list
    
    def run_analysis(self):    
           
        self.read_data()
        self.create_dict_mouse_day_df() # saved in the object.df_evetcode
        
        self.sort_date() # sort the dates in an ascending order; saved in object.sorted_all_days
        
        #print(Mouse_C17.sorted_all_days)
        is_del = input('delete anything? T or F:')
        if is_del == 'T':
            
            del_dates = eval(input('delete days:')) ###
            self.delete_date(del_dates) # dates in object.sorted_all_days will be deleted
        
        self.training_days = eval(input('Please select training days:'))
        
        self.degradation_days = eval(input('Please select degradation days:'))
        
        self.convert_event_to_trial_df()
        
        self.add_islicking_for_selected()
        self.generate_Stats(every_n_trials = 20)
    def generate_trial_type_degradation(self):
        for day in self.degradation_days:
            df = self.df_trials[day]
            trial_type = []
            for index, row in df.iterrows():
                if len(row['goodor_on']) != 0:
                    if len(row['water_on']) == 0:
                        trial_type.append('cont no reward')
                    else:
                        trial_type.append('cont reward')
                else:
                    if len(row['water_on']) == 0:
                        trial_type.append('noncont no reward')
                    else:
                        trial_type.append('noncont reward')
                        
            self.df_trials[day]['trial_type'] = trial_type          
                
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%%  
    
def colorFader(c1,c2,mix=20): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def draw_5_in_1(df, mouse_id, save = False):
    fig,ax = plt.subplots(nrows=5, ncols=1, sharex=True,figsize = (10,50))
    index1 = 0
    c1='#F30021' #red
    c2='#FFA100' #yellow
    c3 = '#5033DD'
    c4 = '#2BF39E'
    c5 = '#5E23A0'
    c6 = '#F4758A'
    c7 = '#271719'
    c8 = '#BFAEB0'
    mix = df.shape[0]
    font = {'size': 7}
    marker = {'markersize':4}
    
    mpl.rc('font', **font)
    mpl.rc('lines', **marker)
    for index, row in df.iterrows():
        index2 = index1 + len(row['licking_aft_odoron'])
        x  = range(index1,index2)
        
        ax[0].plot(x, row['df_hit'],'p-',linewidth = 3,alpha = 0.8,color=colorFader(c1,c2,(index+1)/mix), label = 'hit' if index == 1 else '')
        ax[1].plot(x, row['df_correj'],'p-',linewidth = 3,alpha = 0.8,color=colorFader(c3,c4,(index+1)/mix),label = 'correct rejection' if index == 1 else '')
        ax[0].legend(loc='lower right',frameon = False)
        ax[1].legend(loc='lower right',frameon = False)
        ax[0].set_ylabel('percentage(%)')
        ax[1].set_ylabel('percentage(%)')
        
        ax[2].plot(x, row['licking_aft_odoron'],'p-',linewidth = 2,alpha = 0.8,color=colorFader(c5,c6,(index+1)/mix))
        ax[2].set_ylabel('# licking \n in action window')
        ax[3].plot(x, row['latency_aft_odoron'],'p-',linewidth = 2,alpha = 0.8,color=colorFader(c5,c6,(index+1)/mix))
        ax[3].set_ylabel('latency to lick\n in action window')
        ax[4].plot(x, row['baselicking'],'p-',linewidth = 2,alpha = 0.8,color=colorFader(c7,c8,(index+1)/mix))
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
    if save:
        save_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        plt.savefig("/Users/lechenqian/OneDrive - Harvard University/2019-data/{}/{}.png".format(mouse_id,save_time), bbox_inches="tight")
    
    plt.show() 
    


#%%
#            
#Mouse_C17 = Mouse('C17')
#Mouse_C17.read_data()
#Mouse_C17.create_dict_mouse_day_df() # saved in the object.df_evetcode
#
#Mouse_C17.sort_date() # sort the dates in an ascending order; saved in object.sorted_all_days
#
##print(Mouse_C17.sorted_all_days)
#is_del = input('delete anything? T or F:')
#if is_del == 'T':
#    
#    del_dates = eval(input('delete days:')) ###
#    Mouse_C17.delete_date(del_dates) # dates in object.sorted_all_days will be deleted
#
#Mouse_C17.training_days = eval(input('Please select training days:'))
#
#Mouse_C17.degradation_days = eval(input('Please select degradation days:'))
#
#df_trials_C17 = Mouse_C17.convert_event_to_trial_df()
#
#df_trials_C17 = Mouse_C17.add_islicking_for_selected()
#Stats_C17 = Mouse_C17.generate_Stats(every_n_trials = 40)
## colorful 5 in 1 plot
#draw_5_in_1(Stats_C17, mouse_id = 'C17', save = False)

#%% C17 analysis
Mouse_C17 = Mouse('C17')
Mouse_C17.run_analysis()

Stats_C17_20 = Mouse_C17.stats['20']
draw_5_in_1(Stats_C17_20, mouse_id = Mouse_C17.mouse_id, save = False)




#%% C19 analysis
Mouse_C19 = Mouse('C19')
Mouse_C19.run_analysis()

Stats_C19_20 = Mouse_C19.stats['20']
draw_5_in_1(Stats_C19_20, mouse_id = Mouse_C19.mouse_id, save = False)

#%% C20 analysis
Mouse_C20 = Mouse('C20')
Mouse_C20.run_analysis()

Stats_C20_20 = Mouse_C20.stats['20']
draw_5_in_1(Stats_C20_20, mouse_id = Mouse_C20.mouse_id, save = False)

#%% Control C18
Mouse_C18 = Mouse('C18')
Mouse_C18.run_analysis()

Stats_C18_20 = Mouse_C18.stats['20']
draw_5_in_1(Stats_C18_20, mouse_id = Mouse_C18.mouse_id, save = False)






#%% C22 analysis
Mouse_C22 = Mouse('C22')
Mouse_C22.run_analysis()

Stats_C22_20 = Mouse_C22.stats['20']
draw_5_in_1(Stats_C22_20, mouse_id = Mouse_C22.mouse_id, save = False)


#%% C13 analysis
Mouse_C13 = Mouse('C13')
Mouse_C13.run_analysis()

Stats_C13_20 = Mouse_C13.stats['20']
draw_5_in_1(Stats_C13_20, mouse_id = Mouse_C13.mouse_id, save = False)

#%% C14 analysis
Mouse_C14 = Mouse('C14')
Mouse_C14.run_analysis()

Stats_C14_20 = Mouse_C14.stats['20']
draw_5_in_1(Stats_C14_20, mouse_id = Mouse_C14.mouse_id, save = False)
#%% Control C21
Mouse_C21 = Mouse('C21')
Mouse_C21.run_analysis()

Stats_C21_20 = Mouse_C21.stats['20']
draw_5_in_1(Stats_C21_20, mouse_id = Mouse_C21.mouse_id, save = False)


#%%
# Let's start plotting something!!

# 1. licking rate in action window for C17, 19, 20

fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True,figsize = (15,10))

save = False
c1='#F30021' #red
c2='#FFA100' #yellow
c3 = '#5033DD'#blue
c4 = '#2BF39E'# green
c5 = '#5E23A0'
c6 = '#F4758A'


#mix = df.shape[0]
font = {'size': 14}
marker = {'markersize':4}

mpl.rc('font', **font)
mpl.rc('lines', **marker)
index1 = 0
for i in range(len(Mouse_C17.licking_actionwindow['40'])):
    index2 = index1+len(Mouse_C17.licking_actionwindow['40'][i])
    x = range(index1,index2)
    mix = len(Mouse_C17.licking_actionwindow['40'])
    
    ax.plot(x, Mouse_C17.licking_actionwindow['40'][i],'p-',linewidth = 2,alpha = 0.2,color=colorFader(c1,c2,(i+1)/mix), label = 'Mouse C17' if i == 1 else '')
    ax.legend(loc='lower right',frameon = False)
    
    ax.set_ylabel('licking rate \n in action window')
   
    index1 = index2
    
index1 = 0    
for i in range(len(Mouse_C19.licking_actionwindow['40'])):
    index2 = index1+len(Mouse_C19.licking_actionwindow['40'][i])
    x = range(index1,index2)
    mix = len(Mouse_C19.licking_actionwindow['40'])
    
    ax.plot(x, Mouse_C19.licking_actionwindow['40'][i],'p-',linewidth = 2,alpha = 0.2,color=colorFader(c3,c4,(i+1)/mix), label = 'Mouse C19' if i == 1 else '')
    ax.legend(loc='lower right',frameon = False)
    
    ax.set_ylabel('licking rate \n in action window')
   
    index1 = index2
    
index1 = 0    
for i in range(len(Mouse_C20.licking_actionwindow['40'])):
    index2 = index1+len(Mouse_C20.licking_actionwindow['40'][i])
    x = range(index1,index2)
    mix = len(Mouse_C20.licking_actionwindow['40'])
    
    ax.plot(x, Mouse_C20.licking_actionwindow['40'][i],'p-',linewidth = 2,alpha = 0.2,color=colorFader(c5,c6,(i+1)/mix), label = 'Mouse C20' if i == 1 else '')
    ax.legend(loc='lower right',frameon = False)
    
    ax.set_ylabel('licking rate \n in action window')
   
    index1 = index2

### mean
mean_licking = []
for value in zip(Mouse_C17.licking_actionwindow['40'], Mouse_C19.licking_actionwindow['40'],Mouse_C20.licking_actionwindow['40']):
    
    temp_list = [sum(j)/3 for j in zip(value[0],value[1],value[2])]
    mean_licking.append(temp_list)

index1 = 0    
for i in range(len(mean_licking)):
    index2 = index1+len(mean_licking[i])
    x = range(index1,index2)
    mix = len(mean_licking)
    if i >=0 and i<7:
        ax.plot(x, mean_licking[i],'p-',linewidth = 2,alpha = 0.9,color=colorFader(c4,c1,(i+1)/(mix-7+1)), label = 'Mean' if i == 1 else '')
    else:
        ax.plot(x, mean_licking[i],'p-',linewidth = 2,alpha = 0.9,color=colorFader(c1,c4,(i-7+1)/(mix+7-len(mean_licking)+1)))
        
    ax.legend(loc='lower right',frameon = False)
    
    ax.set_ylabel('licking rate \n in action window')
   
    index1 = index2


ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)    

plt.xticks([])
if save:
    save_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    plt.savefig("/Users/lechenqian/OneDrive - Harvard University/2019-data/{}/{}.png".format('Analysis',save_time), bbox_inches="tight")

plt.show() 
   
        


#%%
Mouse_list = [Mouse_C13,Mouse_C14,Mouse_C17,Mouse_C18,Mouse_C19,Mouse_C20,Mouse_C21,Mouse_C22]
for mouse in Mouse_list:
    mouse.generate_Stats(every_n_trials = 40)
    
#%%
##### 2. let's plot latency figure!! for C17,19 and 20
    
# 1. licking rate in action window for C17, 19, 20

fig,ax = plt.subplots(nrows=1, ncols=1, sharex=True,figsize = (15,10))

save = False
c1='#F30021' #red
c2='#FFA100' #yellow
c3 = '#5033DD'#blue
c4 = '#2BF39E'# green
c5 = '#5E23A0'
c6 = '#F4758A'


#mix = df.shape[0]
font = {'size': 14}
marker = {'markersize':4}

mpl.rc('font', **font)
mpl.rc('lines', **marker)
index1 = 0
for i in range(len(Mouse_C17.licking_latency['40'])):
    index2 = index1+len(Mouse_C17.licking_latency['40'][i])
    x = range(index1,index2)
    mix = len(Mouse_C17.licking_latency['40'])
    
    ax.plot(x, Mouse_C17.licking_latency['40'][i],'p-',linewidth = 2,alpha = 0.2,color=colorFader(c1,c2,(i+1)/mix), label = 'Mouse C17' if i == 1 else '')
    ax.legend(loc='lower right',frameon = False)
    
    ax.set_ylabel('licking latency \n in action window')
   
    index1 = index2
    
index1 = 0    
for i in range(len(Mouse_C19.licking_latency['40'])):
    index2 = index1+len(Mouse_C19.licking_latency['40'][i])
    x = range(index1,index2)
    mix = len(Mouse_C19.licking_latency['40'])
    
    ax.plot(x, Mouse_C19.licking_latency['40'][i],'p-',linewidth = 2,alpha = 0.2,color=colorFader(c3,c4,(i+1)/mix), label = 'Mouse C19' if i == 1 else '')
    ax.legend(loc='lower right',frameon = False)
    
    ax.set_ylabel('licking latency \n in action window')
   
    index1 = index2
    
index1 = 0    
for i in range(len(Mouse_C20.licking_latency['40'])):
    index2 = index1+len(Mouse_C20.licking_latency['40'][i])
    x = range(index1,index2)
    mix = len(Mouse_C20.licking_latency['40'])
    
    ax.plot(x, Mouse_C20.licking_latency['40'][i],'p-',linewidth = 2,alpha = 0.2,color=colorFader(c5,c6,(i+1)/mix), label = 'Mouse C20' if i == 1 else '')
    ax.legend(loc='lower right',frameon = False)
    
    ax.set_ylabel('licking latency \n in action window')
   
    index1 = index2

### mean
mean_licking = []
std_licking = []
for value in zip(Mouse_C17.licking_latency['40'], Mouse_C19.licking_latency['40'],Mouse_C20.licking_latency['40']):
    
    temp_list = [sum(j)/3 for j in zip(value[0],value[1],value[2])]
    mean_licking.append(temp_list)
    std_licking.append(np.std(value,0))
    

index1 = 0    
for i in range(len(mean_licking)):
    index2 = index1+len(mean_licking[i])
    x = range(index1,index2)
    mix = len(mean_licking)
    if i >=0 and i<7:
        ax.errorbar(x, mean_licking[i], std_licking[i]/2,color=colorFader(c4,c1,(i+1)/(mix-7+1)))
        ax.plot(x, mean_licking[i],'p-',linewidth = 2,alpha = 0.9,color=colorFader(c4,c1,(i+1)/(mix-7+1)), label = 'Mean' if i == 1 else '')
    else:
        ax.errorbar(x, mean_licking[i], std_licking[i]/2,color=colorFader(c1,c4,(i-7+1)/(mix+7-len(mean_licking)+1)))
        ax.plot(x, mean_licking[i],'p-',linewidth = 2,alpha = 0.9,color=colorFader(c1,c4,(i-7+1)/(mix+7-len(mean_licking)+1)))
        
    ax.legend(loc='lower right',frameon = False)
    
    ax.set_ylabel('licking latency \n in action window')
   
    index1 = index2


ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)    

plt.xticks([])
if save:
    save_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    plt.savefig("/Users/lechenqian/OneDrive - Harvard University/2019-data/{}/{}.png".format('Analysis',save_time), bbox_inches="tight")

plt.show() 
   

    
      
#%%%
# yeah, we get to the event plot part, which is easy!   
mouse = Mouse_C20
for day in mouse.degradation_days:
            df = mouse.df_trials[day]
            trial_type = []
            for index, row in df.iterrows():
                if len(row['go_odor_on']) != 0:
                    if len(row['water_on']) == 0:
                        trial_type.append('cont no reward')
                    else:
                        trial_type.append('cont reward')
                else:
                    if len(row['water_on']) == 0:
                        trial_type.append('noncont no reward')
                    else:
                        trial_type.append('noncont reward')
                        
            mouse.df_trials[day]['trial_type'] = trial_type          
       

hahagou = {}
xixizhu = {}
for day in mouse.degradation_days:
    old_type = ''
    licking_contrew_after_contrew =    []
    index_contrew_after_contrew =      []
    licking_contrew_after_noncontrew = []
    index_contrew_after_noncontrew =   []
    licking_prior_cont =               []
    licking_prior_noncont =            []
    
    for index, row in mouse.df_trials[day].iterrows(): 
        current_type = row['trial_type']
        if old_type == 'cont reward' and current_type == 'cont reward':
            x = [i for i in row['licking']  if i> row['go_odor_off'][0] and i< row['go_odor_off'][0]+2.5]
            licking_contrew_after_contrew.append(len(x))
            index_contrew_after_contrew.append(index)
            try:
                prior_old_x = [i for i in mouse.df_trials[day].iloc[index-2,:]['licking']  if i> 2.5 and i< 6]
                licking_prior_cont.append(len(prior_old_x))
            except:
                licking_prior_cont.append(np.nan)
                    
        
        
        elif old_type == 'noncont reward' and current_type == 'cont reward':
            x = [i for i in row['licking']  if i> row['go_odor_off'][0] and i< row['go_odor_off'][0]+2.5]
            licking_contrew_after_noncontrew.append(len(x))
            index_contrew_after_noncontrew.append(index)
            try:
                prior_old_x = [i for i in mouse.df_trials[day].iloc[index-2,:]['licking']  if i> 2.5 and i< 6]
                licking_prior_noncont.append(len(prior_old_x))
            except:
                licking_prior_noncont.append(np.nan)
                
        old_type = current_type
        
    hahagou[day] = [index_contrew_after_contrew,licking_contrew_after_contrew,licking_prior_cont]   
    xixizhu[day] = [index_contrew_after_noncontrew,licking_contrew_after_noncontrew,licking_prior_noncont]
       

#%%


mix = len(mouse.degradation_days)
x = 0
plt.figure(figsize=(10,5))

for i, day in enumerate(mouse.degradation_days):
    mean_cont_after_cont = np.mean(hahagou[day][1])
    std_cont_after_cont  = np.std(hahagou[day][1])
    mean_cont_after_noncont = np.mean(xixizhu[day][1])
    std_cont_after_noncont  = np.std(xixizhu[day][1])
    mean_cont = [mean_cont_after_cont,mean_cont_after_noncont]
    std_cont = [std_cont_after_cont,std_cont_after_noncont]
    #plt.errorbar(x,mean_cont,std_cont,marker = 'p',color=colorFader(c1,c2,(i+1)/(mix+1)),alpha = 0.5)
    plt.plot(x+0.25,mean_cont[0], 'p-',color=c1,alpha = 0.5)
    plt.plot(x+0.5,mean_cont[1], 'p-',color=c2,alpha = 0.5)
    plt.plot([x+0.25,x+0.5],mean_cont, 'p-',color='grey',alpha = 0.5) 
    
    mean_prior_cont = np.mean(hahagou[day][2])
    std_prior_cont  = np.std(hahagou[day][2])
    mean_prior_noncont = np.mean(xixizhu[day][2])
    std_prior_noncont  = np.std(xixizhu[day][2])
    mean = [mean_prior_cont,mean_prior_noncont]
    std = [std_prior_cont,std_prior_noncont]
    #plt.errorbar(x,mean,std,marker = 'p',color=colorFader(c3,c4,(i+1)/(mix+1)),alpha = 0.5)
    plt.plot(x+1,mean[0], 'p-',color=c3,alpha = 0.5) 
    plt.plot(x+1.25,mean[1], 'p-',color=c4,alpha = 0.5) 
    plt.plot([x+1,x+1.25],mean, 'p-',color='grey',alpha = 0.5) 
    x += 3
    plt.xlim([0,15])
    plt.title('C20')
plt.show()
        
        
    
    
    
    
#%%

# good good, let's do the trajectory plot
x =np.array(range(mix))

x












#%%%
# first licking after non-contingent reward

def first_licking_after_noncontrew(Mice):
    
    #initalization
    list_firstlicking = []
    list_priorlicking = []
    remainingtime = 0
    looknext = 0

    df_all = Mice.df_trials
    degradation_days = Mice.degradation_days
    
    for day in degradation_days:
        df = df_all[day]

        for index, row in df.iterrows():
            if looknext == 1:
                try:
                    first_licking = min(row['licking'])
                    list_firstlicking.append(remainingtime+first_licking)
                except:
                    list_firstlicking.append(np.nan)
                looknext = 0
                remainingtime = 0
                
            
            if len(row['go_odor_on']) ==0 and len(row['water_on']) !=  0:
                watertime  = row['water_on'][0]
    
                x = [i for i in row['licking']  if i> watertime]
                y = [i for i in row['licking']  if i< watertime]
                try:
                    first_post_licking = min(x)
                    
                    list_firstlicking.append(first_post_licking-watertime)
                except ValueError:
                    looknext = 1
                    remainingtime = row['trial_end'][0]-row['water_on'][0]
                try:
                    last_prior_licking = max(y)
                    list_priorlicking.append(-watertime+last_prior_licking)
                except:
                    list_priorlicking.append(np.nan)
                
            
    return list_firstlicking,list_priorlicking
#%% continue
first_licking,prior_licking= first_licking_after_noncontrew(Mouse_C20)

plt.figure(figsize = (5,5))
bins1 = np.linspace(0,12,100)
bins2 = np.linspace(-12,0,100)
plt.hist(first_licking,bins1,color = 'tomato')
plt.hist(prior_licking,bins2,color = 'grey')
plt.title('C20')
plt.show()

plt.figure(figsize = (25,5))
x = range(len(first_licking))
plt.scatter(x,first_licking,color = 'tomato', marker='o' )

plt.scatter(x,prior_licking,color='grey', marker='o' )
for a in zip(x,first_licking,prior_licking):   
    plt.plot([a[0],a[0]], [a[2],a[1]],color = 'grey',alpha = 0.5)
plt.scatter(x,first_licking,color = 'tomato', marker='o' )
plt.scatter(x,prior_licking,color='grey', marker='o' )
plt.title('C20')
plt.show()
#%%

c1='#F30021' #red
c2='#FFA100' #yellow
c3 = '#5033DD'
c4 = '#2BF39E'
c5 = '#5E23A0'
c6 = '#F4758A'
c7 = '#271719'
c8 = '#BFAEB0'   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
