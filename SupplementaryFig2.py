#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 09:38:14 2025

@author: joannafernandez

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os 

#import data file and name cols 
def Find(file):
    genes = pd.read_csv(file, delimiter=",")
    genes.rename(columns={'Unnamed: 0': 'replicate'}, inplace=True)
    genes.rename(columns={'Unnamed: 1': 'indels'}, inplace=True)

    return genes

R0perc = Find("/path/to/tideR0_percentages.csv")
R0perc= R0perc.dropna() #remove empty rows
R0pval = Find("/path/to/Downloads/tideR0_pvals.csv")
R0pval= R0pval.dropna() #remove empty rows

#where p>0.001 (non-significant indel) remove from consideration
R0perc.loc[:, ~R0perc.columns.isin(['replicate', 'indels'])] = R0perc.loc[:, ~R0perc.columns.isin(['replicate', 'indels'])].mask(R0pval > 0.001, 0)

#formating for plotting
melted_df = R0perc.melt(id_vars=['replicate', 'indels'],
                               var_name='condition',
                               value_vars= ['sgLAD_6hr',
                                      'sgLAD_24hr', 'sgLAD_48hr'],
                               value_name='percentage')
melted_df['percentage'] = melted_df['percentage'].astype(int)

#%%
#assign indel type 
def classify_indel(indel):
    if indel == 0:
        return "uncut"
    elif -4 <= indel <= 2:
        return "NHEJ"
    elif -20 <= indel <= -3:
        return "MMEJ"
    else:
        return None


melted_df['class'] = melted_df['indels'].apply(classify_indel)
melted_df = melted_df[melted_df['class'].notna()]

#find total % of sequences that are either NHEJ or MMEJ. (uncut/wt sequences are only ever "0")
summed_df = (
    melted_df
    .groupby(['condition', 'replicate', 'class'], as_index=False)
    .agg(summed_percentage=('percentage', 'sum'))
)


subset = summed_df.copy()

#Calculate total NHEJ + uncut per condition/replicate
total = subset.groupby(['condition', 'replicate'])['summed_percentage'].transform('sum')
#Calculate percentage within that total
subset['relative_percentage'] = (subset['summed_percentage'] / total) * 100
# Split 'condition' into two new columns: 'condition' and 'time'
subset[['condition', 'time']] = subset['condition'].str.split('_', n=1, expand=True)



#%%

hue_order = ['6hr', '24hr', '48hr']

custom_palette = {
    'MMEJ': '#D33873',
    'NHEJ': '#575757',
    'uncut': '#EFB54F'
}

'''
sns.set_context("paper", font_scale=2.5)


g = sns.catplot(
    data=subset,
    x="class",
    y="relative_percentage",
    hue="class",
    col="time",
    kind="bar",
    palette=custom_palette,
    col_order=hue_order,
    errorbar= None,
    height=4,
    aspect=1.8,
    dodge=False,
    legend=False,
    alpha = 0.9
)

for ax, cond in zip(g.axes.flat, hue_order):
    data = subset[subset['time'] == cond]
    sns.stripplot(
        data=data,
        x="class",
        y="relative_percentage",
        ax=ax,
        palette=["white"],  
        alpha=0.8,
        size=14,
        edgecolor='black',
        linewidth=0.5,
        dodge=False,
        order=["MMEJ", "NHEJ", "uncut"]
    )

g.set_axis_labels("", "% of sequences")
g.set_titles("{col_name}")
plt.tight_layout()
plt.show()
'''

g = sns.catplot(
    data=subset,
    x="class",
    y="relative_percentage",
    hue="class",
    col="time",  
    row="condition",
    kind="bar",
   # errorbar=("se", 1),
    capsize=0.25,  
    width=0.7,  
    palette=custom_palette,
    col_order=hue_order,

    height=4,
    errorbar=None,
    aspect=1.8,
    dodge=False,
    legend=False,
    alpha=0.9,
)

for (row_val, col_val), ax in g.axes_dict.items():
    subsety = subset[(subset['condition'] == row_val) & (subset['time'] == col_val)]

    sns.swarmplot(
        data=subsety,
        x="class",
        y="relative_percentage",
        color="white",
        edgecolor="black",
        linewidth=1.5,
        size=10,
        dodge=False,
        ax=ax,
        zorder=4, clip_on=False
    )

    ax.axhline(y=50, color='black', linestyle='--', linewidth=1)
    

for ax in g.axes.flat:
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', which='both', labelleft=True)  
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', which='both', labelbottom=True)

    ax.axhline(50, linestyle='--', color='grey', linewidth=1)


for ax in g.axes.flat:
    for p in ax.patches:
        p.set_edgecolor("black")
        p.set_linewidth(1.5)
        p.set_joinstyle("miter")


for ax in g.axes.flat:

    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='both', width=1.6, length=4)

g.set_xlabels(fontsize=10)
g.set_ylabels(fontsize=10)
g.set_titles(size=10)

for ax in g.axes.flat:
    ax.tick_params(axis='y', labelsize=22)

sns.despine()
plt.tight_layout()
plt.show()   

#plt.savefig("bypathwayrelative.svg", dpi=300, bbox_inches='tight')



plt.savefig("sup2g.svg", dpi=300, bbox_inches='tight')  # Or .pdf/.png



