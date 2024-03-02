import pandas as pd
import numpy as np
#import os 
# make it a class with different functions doing different things 
# path = './Project_NbSe2_code/data/pdphasegammapkt.csv'
class DataProcessing:
    def __init__(self, astart, alast, astep,path):
        self.astart = astart
        self.alast = alast
        self.astep = astep
        self.path = path


    def phase_bdd(self):
        df = pd.read_csv(self.path, header=None)
        lss = []
        for ai in np.arange(self.astart,self.alast,self.astep): 
            print('ai = ',ai)
            ls = []
            for col in range(len(df[0])):
                #print((df[0][col] - ai))
                if (df[0][col] - ai)<10**(-12):
                    #print('diff = ',(df[0][col] - ai))
                    ls.append(df[1][col])
                    #print(ls)        
                else:
                    pass  
            m = max(ls)
            lss.append((ai,m))   
        #print(lss)
        return lss
    
    #def phase_bdd_pd_with_allpkts():





 
