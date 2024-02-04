# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:47:19 2023

@author: kikih
"""
import os
import pandas as pd
import numpy as np
import pytensor.tensor as pt
import seaborn as sns
#import pymc as pm # alleen nodig voor simuleren
import pingouin as pg
import matplotlib.pyplot as plt
from pathlib import Path
from metadpy import sdt
from metadpy.plotting import plot_confidence
from metadpy.utils import discreteRatings, trials2counts
from scipy.stats import norm, sem, t 
#from systole.detection import ppg_peaks
#import graphviz
#import arviz as az

sns.set_context('talk')

## set directories
os.chdir('C:/Users/kikih/OneDrive - Universiteit Utrecht/Studie/Neuroscience and Cognition/Major internship - body image and hormones/data hrd') #set working directory
reportPath = Path("C:/Users/kikih/Documents/reports") # hier sla je data in op
outputpath = Path(Path.cwd(), "output") # hier haal je data uit
dataPathHC = Path(os.path.join(reportPath, "HC")) #hier sla je de data van de HC groep op
dataPathn = Path(os.path.join(reportPath, "NC")) #hier sla je data van de NC groep op
resultPath = Path(Path.cwd(), "data")

###########################
#### Functions ############
###########################
def cumulative_normal(x, alpha, beta, s=np.sqrt(2)):
    # Cumulative distribution function for the standard normal distribution
    return 0.5 + 0.5 * pt.erf((x - alpha) / (beta * s))

def reversals(df):
    d = np.diff(df.Alpha.to_numpy())
    reversals = d[:-1] != d[1:]
    return np.median(df.Alpha.to_numpy()[1:-1][reversals])

def plot_psychometricFunctions(df, title):

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    for i, modality, col in zip((0, 1), ['Intero', 'Extero'], ['#c44e52', '#4c72b0']):
        threshold, slope = [], []
        for subject in df.Subject.unique():
            threshold.append(df.Threshold[(behavior_df.Modality == modality) & (behavior_df.Subject == subject)].values)
            slope.append(df.Slope[(behavior_df.Modality == modality) & (behavior_df.Subject == subject)].values)

            # Plot Psi estimate of psychometric function
            axs[i].plot(np.linspace(-40, 40, 500), 
                    (norm.cdf(np.linspace(-40, 40, 500), loc=threshold[-1], scale=slope[-1])),
                    '-', color='gray', alpha=.05)
            axs[i].set_ylabel('P$_{(Response=Faster)}$', size=12)
            axs[i].set_xlabel('Intensity ($\Delta$ BPM)', size=12)
        axs[i].plot(np.linspace(-40, 40, 500), 
                (norm.cdf(np.linspace(-40, 40, 500), loc=np.array([threshold]).mean(), scale=np.array([slope]).mean())),
                '-', color=col, linewidth=4)
        axs[i].axvline(x=np.array([threshold]).mean(), ymin=0, ymax=0.5, linestyle='--', color=col, linewidth=2)
        axs[i].plot(np.array([threshold]).mean(), 0.5, 'o', color=col, markersize=15)
        axs[i].plot(np.array([threshold]).mean(), 0.5, 'o', color='w', markersize=12)
        axs[i].minorticks_on()
        axs[i].set_axisbelow(True)
        axs[i].grid(which='major', linestyle='--')
        axs[i].tick_params(axis='y', which='minor', left=False)

    plt.tight_layout()
    sns.despine()
    #plt.savefig(path + f'/figures/{title}.svg', dpi=600)
   
# define subjects
subjectshc = os.listdir(dataPathHC)
subjectsn = os.listdir(dataPathn)

# load data per group
session = 'HC'
mergedHC_df = pd.DataFrame([])
for sub in subjectshc:
    df = pd.read_csv(os.path.join(dataPathHC, sub))
    df["Subject"] = sub
    df = df.rename(columns={'Estimation': 'Decision', 'EstimationRT': 'DecisionRT'})
    mergedHC_df = mergedHC_df.append(df, ignore_index=True)
    
remove = []
for sub_rm in remove:
    mergedHC_df = mergedHC_df[mergedHC_df.Subject != sub_rm]

mergedHC_df = mergedHC_df.dropna(axis='index')

# Save data frame HC
mergedHC_df.to_csv(
    os.path.join(
        outputpath, 'HC_merged.txt'),
    index=False)    
print(f'{mergedHC_df.Subject.nunique()} participants use HC')

session = 'N'
mergedN_df = pd.DataFrame([])
for sub in subjectsn:
    dfN = pd.read_csv(os.path.join(dataPathn, sub))
    dfN["Subject"] = sub
    dfN = dfN.rename(columns={'Estimation': 'Descision', 'EstimationRT': 'DecisionRT'})
    mergedN_df = mergedN_df.append(dfN, ignore_index=True)
    
remove = []
for sub_rm in remove:
    mergedN_df = mergedN_df[mergedN_df.Subject != sub_rm]

mergedN_df = mergedN_df.dropna(axis='index')

# Save data frame
mergedN_df.to_csv(
    os.path.join(
        outputpath, 'N_merged.txt'),
    index=False)    
print(f'{mergedN_df.Subject.nunique()} participants do not use HC')

###########################
#### Summary dataframe ####
###########################

### psychometric parameters are estimated here using the merged files
behavior_df = pd.DataFrame([])
for session in ['HC', 'N']:
    group_df = pd.read_csv(
        os.path.join(outputpath, f'{session}_merged.txt'))
 
    for sub in group_df['Subject'].unique():
        taskDuration = (
            group_df.loc[group_df.Subject == sub, "StartListening"].to_numpy()[-1] -
                group_df.loc[group_df.Subject == sub, "StartListening"].to_numpy()[0]
            ) / 60

        for modality in ["Intero", "Extero"]:            
            threshold, slope, decisionRT, confidenceRT, accuracy, confidence,\
                threshold_updown = None, None, None, None, None, None, None
            
            this_df = group_df[(group_df.Subject == sub) & (group_df.Modality == modality)]

            threshold, slope = (
                this_df[~this_df.EstimatedThreshold.isnull()].EstimatedThreshold.iloc[-1],
                this_df[~this_df.EstimatedSlope.isnull()].EstimatedSlope.iloc[-1],
            )
            decisionRT, confidenceRT = (
                this_df["DecisionRT"].median(),
                this_df.ConfidenceRT.median(),
            )
            accuracy, confidence = (
                this_df["ResponseCorrect"].mean() * 100,
                this_df["Confidence"].mean(),
            )

            # Threshold from up/down staircase
            threshold_updown = np.mean([
                reversals(this_df[this_df.StairCond=='low']),
                reversals(this_df[this_df.StairCond=='high'])
            ])

            # Ratio of staircase corruption
            ratio = ((this_df.Alpha - this_df.EstimatedThreshold).loc[-40:] > 0).mean()
            corruption_ratio = np.abs(ratio-0.5)*2

            behavior_df = behavior_df.append(
                {
                    "Subject": sub,
                    "Session": session,
                    "Modality": modality,
                    "Accuracy": accuracy,
                    "Confidence": confidence,
                    "Threshold": threshold,
                    "Threshold_UpDown": threshold_updown,
                    "Slope": slope,
                    "TaskDuration": taskDuration,
                    "DecisionRT": decisionRT,
                    "ConfidenceRT": confidenceRT,
                    "Corruption": corruption_ratio,
                },
                ignore_index=True,
            )

# Save data frame
behavior_df.to_csv(
    os.path.join(
        outputpath, 'behavior.txt'),
    index=False)

###########################
#### TASK DURATION ########
###########################
sns.set_context('talk')
plt.figure(figsize=(8, 5))
sns.histplot(data=behavior_df[behavior_df.Modality=='Intero'],
             x='TaskDuration', hue='Session')
plt.xlabel('Time (min)')
plt.ylabel('HRD')
sns.despine()

for session in ['HC', 'N']:
    print(f'Session: {session} - Task mean time: {round(behavior_df[behavior_df.Session == session].TaskDuration.mean(), 2)} min')
    print(f'Session: {session} - Task std time: {round(behavior_df[behavior_df.Session == session].TaskDuration.std(), 2)} min')
    print(f'Session: {session} - Task max time: {round(behavior_df[behavior_df.Session == session].TaskDuration.max(), 2)} min')
    print(f'Session: {session} - Task min time: {round(behavior_df[behavior_df.Session == session].TaskDuration.min(), 2)} min')

###########################
#### Psi convergence index 
###########################
sns.set_context('notebook')
fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharey='row')

sns.stripplot(data=behavior_df, x='Session', y='Corruption', hue='Modality', dodge=1, edgecolors='k', palette=['#c44e52', '#4c72b0'], linewidths=1, alpha=0.3, ax=ax)
sns.barplot(data=behavior_df, x='Session', y='Corruption', hue='Modality', alpha=0.2, palette=['#c44e52', '#4c72b0'], ax=ax)

plt.xticks([0, 1], ['Session 1', 'Session 2'], size=12)
plt.xlabel('')
ax.minorticks_on()
ax.set_axisbelow(True)
ax.grid(which='major', axis='y',linestyle='--')
ax.tick_params(axis='x', which='minor', bottom=False)
ax.set_ylabel('Index of incomplete convergence', size=15)
ax.set_ylim([0, 1])
plt.tight_layout()
sns.despine()
# plt.savefig(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + f'/figures/SupplementaryMaterial2Stripplot.svg', dpi=300)

intero = (behavior_df[(behavior_df.Session=='HC') & (behavior_df.Modality=='Intero')].Corruption > 0.5).sum()
extero = (behavior_df[(behavior_df.Session=='HC') & (behavior_df.Modality=='Extero')].Corruption > 0.5).sum()
print(f'HC: Intero: {intero} and Extero: {extero} staircases with a high incomplete convergence index.')
intero = (behavior_df[(behavior_df.Session=='N') & (behavior_df.Modality=='Intero')].Corruption > 0.5).sum()
extero = (behavior_df[(behavior_df.Session=='N') & (behavior_df.Modality=='Extero')].Corruption > 0.5).sum()
print(f'N: Intero: {intero} and Extero: {extero} staircases with a high incomplete convergence index.')
 
##################################################
##### Psychometric parameters - Psi estimates ####
##################################################
behavior_df = pd.read_csv(os.path.join(outputpath, 'behavior.txt'))
print(f'n HC = {behavior_df[(behavior_df.Session=="HC") & (~behavior_df["Slope"].isnull())].Subject.nunique()}')
plot_psychometricFunctions(
    df=behavior_df[(behavior_df.Session=='HC') & (~behavior_df['Slope'].isnull()) & (~behavior_df['Threshold'].isnull())],
    title='HC psi')

threshHC = pg.pairwise_tests(data=behavior_df[behavior_df.Session=='HC'], dv='Threshold', within='Modality', subject='Subject', effsize='cohen')
slopeHC = pg.pairwise_tests(data=behavior_df[behavior_df.Session=='HC'], dv='Slope', within='Modality', subject='Subject', effsize='cohen')

# natural cycle
print(f'n NC = {behavior_df[(behavior_df.Session=="N") & (~behavior_df["Slope"].isnull())].Subject.nunique()}')
plot_psychometricFunctions(
    df=behavior_df[(behavior_df.Session=='N') & (~behavior_df['Slope'].isnull()) & (~behavior_df['Threshold'].isnull())],
    title='N')

threshNC = pg.pairwise_tests(data=behavior_df[behavior_df.Session=='N'], dv='Threshold', within='Modality', subject='Subject', effsize='cohen')
slopeNC = pg.pairwise_tests(data=behavior_df[behavior_df.Session=='N'], dv='Slope', within='Modality', subject='Subject', effsize='cohen')

###########################
#### estimates  HC ########
###########################
'''
HC = pd.read_csv(os.path.join(outputpath, 'HC_psychophysics.txt'))
print(f'n Session 1 = {HC.Subject.nunique()}')

 # threshold hc
HCthreshInt = pg.compute_bootci(x=HC[HC.Modality=='Intero'].BayesianThreshold.to_numpy(), func='mean')
HCthreshExt = pg.compute_bootci(x=HC[HC.Modality=='Extero'].BayesianThreshold.to_numpy(), func='mean')
threshHCTT = pg.pairwise_tests(data=HC, subject='Subject', dv='BayesianThreshold', within='Modality', effsize='cohen', return_desc=True)

 # slopes hc
HCslopeInt = pg.compute_bootci(x=HC[HC.Modality=='Intero'].BayesianSlope.to_numpy(), func='mean')
HCslopeExt = pg.compute_bootci(x=HC[HC.Modality=='Extero'].BayesianSlope.to_numpy(), func='mean')
HCslopeTT = pg.pairwise_tests(data=HC, subject='Subject', dv='BayesianSlope', within='Modality', effsize='cohen', return_desc=True)

###########################
#### estimates NC #########
###########################
NC = pd.read_csv(os.path.join(outputpath, 'N_psychophysics.txt'))
print(f'n Session 2 = {NC.Subject.nunique()}')

plot_psychometricFunctions(
    df=NC,
    title='NC')

NCthreshInt = pg.compute_bootci(x=NC[NC.Modality=='Intero'].BayesianThreshold.to_numpy(), func='mean')
NCthreshExt = pg.compute_bootci(x=NC[NC.Modality=='Extero'].BayesianThreshold.to_numpy(), func='mean')
NCthreshTT = pg.pairwise_tests(data=NC, subject='Subject', dv='BayesianThreshold', within='Modality', effsize='cohen', return_desc=True)

 # slopes nc
NCslopeInt = pg.compute_bootci(x=NC[NC.Modality=='Intero'].BayesianSlope.to_numpy(), func='mean')
NCslopeExt = pg.compute_bootci(x=NC[NC.Modality=='Extero'].BayesianSlope.to_numpy(), func='mean')
NCslopeTT = pg.pairwise_tests(data=NC, subject='Subject', dv='BayesianSlope', within='Modality', effsize='cohen', return_desc=True)
'''
###########################
#### META COGNITION #######
###########################
sns.set_context('notebook')

metacognition_df = pd.DataFrame([])
drop = []
for session in ['HC', 'N']:
    merged = pd.read_csv(
        os.path.join(outputpath, f'{session}_merged.txt'))

    for sub in merged['Subject'].unique():
        
        for modality in ['Extero', 'Intero']:
            this_df = merged[(merged.Subject==sub) & (merged.Modality==modality)]

            # Drop NAs
            this_df = this_df[~this_df.Confidence.isnull()]
            
            # Remove HR outliers
            this_df = this_df[this_df.HeartRateOutlier==0]
            
            # Discretize ratings
            try:
                this_df.loc[:, 'ConfidenceRaw'] = this_df.Confidence.to_numpy()
                new_ratings, out = discreteRatings(this_df.Confidence.to_numpy(), verbose=False)
                this_df.loc[:, 'Confidence'] = new_ratings
                this_df['Session'] = session
                this_df['Stimuli'] = this_df['Alpha'] > 0
                this_df['Responses'] = this_df['Decision'] == 'More'
                this_df['Accuracy'] = (this_df['Stimuli'] & this_df['Responses']) | (~this_df['Stimuli'] & ~this_df['Responses'])
                this_df = this_df[['Subject', 'Session', 'Modality', 'Stimuli', 'Accuracy', 'Responses', 'Confidence', 'ConfidenceRaw']]
                metacognition_df = metacognition_df.append(this_df, ignore_index=True)
                #if (this_df['Stimuli'].sum() == 0):
                 #  drop.append(sub)
            except ValueError:
                print(f'Dropping subject {sub} due to invalid ratings - Session: {session}')
                drop.append(sub)
                
    for sub in drop:
        metacognition_df = metacognition_df[metacognition_df.Subject!=sub]

metacognition_df.to_csv(
    os.path.join(
        outputpath, 'metacognition_trials.txt'),
    index=False)

sdt_df = pd.DataFrame([])
responsesRatings_df = pd.DataFrame([])
for session in ['HC', 'N']:
    for sub in metacognition_df[metacognition_df.Session==session].Subject.unique():
        for cond in metacognition_df.Modality.unique():
            this_df = metacognition_df[(metacognition_df.Subject==sub) & (metacognition_df.Session==session) & (metacognition_df.Modality==cond)]
                    
            nR_S1, nR_S2 = trials2counts(
                data=this_df, stimuli='Stimuli', accuracy='Accuracy',
                confidence='Confidence', nRatings=4)

            responsesRatings_df = responsesRatings_df.append(pd.DataFrame({
                'Subject': sub, 'Modality': cond, 'Session': session, 'nR_S1': nR_S1, 'nR_S2': nR_S2}))

            sdt_df = sdt_df.append(pd.DataFrame({
                'Subject': [sub],
                'Session': session,
                'Modality': [cond],
                'RespCountS1': nR_S1.sum(),
                'RespCountS2': nR_S2.sum(),
                'Accuracy': this_df.Accuracy.to_numpy().mean() * 100,
                'ConfidenceRaw': this_df.ConfidenceRaw.to_numpy().mean(),
                'd': [this_df.dprime()],
                'c': [this_df.criterion()]}))

for sub in drop:
    responsesRatings_df = responsesRatings_df[responsesRatings_df.Subject!=sub]
    sdt_df = sdt_df[sdt_df.Subject!=sub]
    
responsesRatings_df.to_csv(os.path.join(
    outputpath, 'responsesRatings.txt'),
    index=False)
sdt_df.to_csv(os.path.join(
    outputpath, 'sdt.txt'),
    index=False)

sdtint_df = sdt_df[sdt_df.Modality == 'Intero']
sdtext_df = sdt_df[sdt_df.Modality == 'Extero']

# mean d' + crit per group and session
sdtint_df.groupby('Session')[['d']].mean() # mean sensitivity interoceptive
sdtint_df.groupby('Session' )[['c']].mean() # response bias

sdtext_df.groupby('Session')[['d']].mean() # mean sensitivity extroceptive
sdtext_df.groupby('Session' )[['c']].mean() # response bias

# criterion: larger = lower willingness to say 'faster' in an ambiguous situation, more conservative | d': sensitivity, larger = more sensitive to difference between faster and slower, near 0 = chance performance

###########################
#### D': sensitvity CI + t-test
###########################
HCdInt = pg.compute_bootci(x=sdt_df[(sdt_df.Session == 'HC') & (sdt_df.Modality == 'Intero')].d.to_numpy(), func='mean')
HCdExt = pg.compute_bootci(x=sdt_df[(sdt_df.Session == 'HC') & (sdt_df.Modality == 'Extero')].d.to_numpy(), func='mean')
HCdTT = pg.pairwise_tests(data=sdt_df[sdt_df.Session == 'HC'], dv='d', within='Modality', subject='Subject', effsize='cohen', return_desc=True)

NCdInt = pg.compute_bootci(x=sdt_df[(sdt_df.Session == 'N') & (sdt_df.Modality == 'Intero')].d.to_numpy(), func='mean')
NCdExt = pg.compute_bootci(x=sdt_df[(sdt_df.Session == 'N') & (sdt_df.Modality == 'Extero')].d.to_numpy(), func='mean')
NCdTT = pg.pairwise_tests(data=sdt_df[sdt_df.Session == 'N'], dv='d', within='Modality', subject='Subject', effsize='cohen', return_desc=True)

###########################
#### CRITERION: bias CI + ttest
###########################
HCcritInt = pg.compute_bootci(x=sdt_df[(sdt_df.Session == 'HC') & (sdt_df.Modality == 'Intero')].c.to_numpy(), func='mean')
HCcritExt = pg.compute_bootci(x=sdt_df[(sdt_df.Session == 'HC') & (sdt_df.Modality == 'Extero')].c.to_numpy(), func='mean')
HCcritTT = pg.pairwise_tests(data=sdt_df[sdt_df.Session == 'HC'], dv='c', within='Modality', subject='Subject', effsize='cohen', return_desc=True)

NCcritInt = pg.compute_bootci(x=sdt_df[(sdt_df.Session == 'N') & (sdt_df.Modality == 'Intero')].c.to_numpy(), func='mean')
NCcritExt = pg.compute_bootci(x=sdt_df[(sdt_df.Session == 'N') & (sdt_df.Modality == 'Extero')].c.to_numpy(), func='mean')
NCcritTT = pg.pairwise_tests(data=sdt_df[sdt_df.Session == 'N'], dv='c', within='Modality', subject='Subject', effsize='cohen', return_desc=True)

###########################
#### raw confidence #######
###########################
rawConfidence_df = pd.DataFrame([])
for session in ['HC', 'N']:
    for sub in metacognition_df[metacognition_df.Session==session].Subject.unique():
        for cond in metacognition_df.Modality.unique():
            for corr in [True, False]:
                this_df = metacognition_df[(metacognition_df.Subject==sub) & (metacognition_df.Modality==cond) & (metacognition_df.Session==session) & (metacognition_df.Accuracy==corr)]
                new_ratings = this_df.Confidence.to_numpy()
                rawConfidence_df = rawConfidence_df.append(pd.DataFrame({
                    'Subject': sub,
                    'Session': session,
                    'Modality': cond,
                    'Correct': corr,
                    'Ratings': np.arange(1, 5),
                    'Density': np.array([np.count_nonzero(new_ratings == i) for i in range(1, 5)]) / len(new_ratings)}), ignore_index=True)
 
###### hc
sns.set_context('talk')
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8,8))
for i, cond in enumerate(['Extero', 'Intero']):
    sns.barplot(data=rawConfidence_df[(rawConfidence_df.Modality==cond) & (rawConfidence_df.Session=='HC')], x='Ratings', y='Density', hue='Correct', ax=axs[i], palette=["#b55d60", "#5f9e6e"])
    labels = [item.get_text() for item in axs[i].get_xticklabels()]

axs[0, 0].set_ylabel('Density', size=15)
axs[0, 1].set_ylabel('')
axs[0, 0].set_xlabel('')
axs[0, 1].set_xlabel('')
axs[0, 0].set_ylim(0, 0.65)
axs[0, 1].set_ylim(0, 0.65)
plt.tight_layout()
sns.despine()
#plt.savefig(path + '/figures/Fig3_metacognition.svg', dpi=300)

samples_dfHC = pd.read_csv(
    os.path.join(reportPath, 'jagsSamples_HC.txt'),sep='\t')
stats_dfHC = pd.read_csv(
    os.path.join(reportPath, 'jagsStats_HC.txt'),sep='\t')
stats_dfHC['Modality'] = 'Intero'
stats_dfHC.loc[stats_dfHC.name.str.contains(",2]"), 'Modality'] = 'Extero'
stats_dfHC = stats_dfHC[stats_dfHC.name.str.contains('Mratio')]
stats_dfHC = stats_dfHC.sort_values('Modality')
MratioHCint = (stats_dfHC[stats_dfHC.Modality=='Intero']['mean'].mean()) # metacognition interoception HC
MratioHCext = (stats_dfHC[stats_dfHC.Modality=='Extero']['mean'].mean()) # metacognition exteroception HC

###### NC

samples_dfNC = pd.read_csv(
    os.path.join(reportPath, 'jagsSamples_NC.txt'),sep='\t')
stats_dfNC = pd.read_csv(
    os.path.join(reportPath, 'jagsStats_NC.txt'),sep='\t')
stats_dfNC['Modality'] = 'Intero'
stats_dfNC.loc[stats_dfNC.name.str.contains(",2]"), 'Modality'] = 'Extero'
stats_dfNC = stats_dfNC[stats_dfNC.name.str.contains('Mratio')]
stats_dfNC = stats_dfNC.sort_values('Modality')
MratioNCint = (stats_dfNC[stats_dfNC.Modality=='Intero']['mean'].mean()) # metacognition interoception NC
MratioNCext = (stats_dfNC[stats_dfNC.Modality=='Extero']['mean'].mean()) # metacognition exteroception NC

Mratio_df = pd.DataFrame(data = (MratioHCint, MratioNCint, MratioHCext, MratioNCext))
Mratio_df.to_csv(
    os.path.join(
        outputpath, 'Mratio.txt'),
    index=False)    
