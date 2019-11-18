# Quantifying-the-scale-effect-using-semi-variograms
（Source code for paper） Quantifying the scale effect in geospatial big data using semi-variograms. https://doi.org/10.1371/journal.pone.0225139
## Abstract：
The scale effect is an important research topic in the field of geography. When aggregating individual-level data into areal units, encountering the scale problem is inevitable. This problem is more substantial when mining collective patterns from big geo-data due to the characteristics
of extensive spatial data. Although multi-scale models were constructed to mitigate this issue, most studies still arbitrarily choose a single scale to extract spatial patterns. In this research, we introduce the nugget-sill ratio (NSR) derived from semi-variograms as an indicator to extract the optimal scale. We conducted two simulated experiments to demonstrate the feasibility of this method. Our results showed that the optimal scale is negatively
correlated with spatial point density, but positively correlated with the degree of dispersion in a point pattern. We also applied the proposed method to a case study using Weibo check-in data from Beijing, Shanghai, Chengdu, and Wuhan. Our study provides a new perspective to measure the spatial heterogeneity of big geo-data and selects an optimal spatial scale for big data analytics.
## Usgae：
* getvariance.py is used to aggregate point data and calculate semi-variance for each grid pair.
* getNSR.py is used to fit empirical variogram and get nugget-sill ratio (NSR).
* WEIBO_CHECKINS_2014 includes POI check-ins data we used in this work.
## Citation
Please cite our paper if this helps you in your own work：

Chen L, Gao Y, Zhu D, Yuan Y, Liu Y (2019) Quantifying the scale effect in geospatial big data using semi-variograms. PLOS ONE 14(11): e0225139.  

@article{Chen, Lei AND Gao, Yong AND Zhu, Di AND Yuan, Yihong AND Liu, Yu},  
    journal = {PLOS ONE},  
    publisher = {Public Library of Science},  
    title = {Quantifying the scale effect in geospatial big data using semi-variograms},  
    year = {2019},  
    month = {11},  
    volume = {14},  
    url = {https://doi.org/10.1371/journal.pone.0225139},  
    pages = {1-18},  
    number = {11},  
    doi = {10.1371/journal.pone.0225139}   
}
