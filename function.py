# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:31:33 2023

@author: kikih
"""
import glob, os
import xarray
import pathlib
import matplotlib
from pathlib import Path
from cardioception.reports import report, preprocessing

os.chdir('C:/Users/kikih/OneDrive - Universiteit Utrecht/Studie/Neuroscience and Cognition/Major internship - body image and hormones/data hrd') #set working directory
reportPath = Path("C:/Users/kikih/Documents/reports")   # Location outside of onedrive
resultPath = Path(Path.cwd(), "data_all", "BOTH", "027") # make sure you have a folder per participant, change the number for every participant
    

# for each file found in the result folder, create the HTML report
for f in resultPath.iterdir():

    # the input is a file name at it returns a summary dataframe
   report(result_path=resultPath, report_path=Path(reportPath), task = "HRD")
