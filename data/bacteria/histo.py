import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns


#with open( './gbsa-result.csv', newline='' ) as csvfile :
#    csvreader = csv.reader( csvfile, delimiter=',', quotechar='|' )

df = pd.read_csv( './chembl28_bacteria_activity.csv' )
df = df.loc[ df['PCHEMBL_VALUE'] > 0, : ]
df = df.filter( items=['PCHEMBL_VALUE']  )
print( df )

def plot_dG( data, title ):
    bins = np.arange( np.floor( min(data) ), np.ceil( max(data) ), 0.5 )
    plt.hist( data, bins, histtype='step' )
#    plt.yscale( 'log' )
    plt.title( title )
    plt.show()
    plt.savefig( 'chembl28_bacteria_activity-histo.png' )

plot_dG( df['PCHEMBL_VALUE'], 'PCHEMBL_VALUE' )

plt.clf()

