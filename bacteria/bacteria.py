import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import pandas as pd
from pandas import concat
from keras import layers, models, losses, metrics
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
from rdkit import DataStructs





def GenerateMoleculeFingerprint( row, nfp=2048 ):
    m = Chem.MolFromSmiles( row.CANONICAL_SMILES )
    info={}
    arr = None
    try:
        fp = Chem.GetMorganFingerprintAsBitVect( m, 2, nBits=nfp, bitInfo=info )
        arr = np.zeros( (1, ) )
        DataStructs.cDataStructs.ConvertToNumpyArray( fp, arr )
    except Exception :
        pass
    return arr


def printNumActivityClass( label, df ) :
    num_active = len( df[ df.activity_class == 1 ].index )
    num_inactive = len( df[ df.activity_class == 0 ].index )
    print( "\n===>%s : activity class = %d + %d / %d\n" %  ( label, num_active, num_inactive, len(df) ) )
    return



class MyDataFrame :
    def __init__( self, fname, nBits=1024, activity_criteria=6.0 ) :
        self.Initialize( fname, nBits, activity_criteria )
        return

    def Initialize( self, fname, nBits=1024, activity_criteria=6.0 ) :
        self.ReadChEMBLCSV( fname )
        self.purge_datafile ()
        self.make_fingerprint(nBits )
        self.make_activityclass(activity_criteria )
        return

    def ReadChEMBLCSV( self, fname ) :
        self.mydf = pd.read_csv( fname, usecols=['PCHEMBL_VALUE', 'CANONICAL_SMILES'], nrows=9999999 )
        # self.mydf = pd.read_csv( fname, delimiter=';', usecols=['pChEMBL Value', 'Smiles'], nrows=9999999 )
        if not set( [ 'PCHEMBL_VALUE', 'CANONICAL_SMILES' ] ).issubset( self.mydf.columns ) :
            self.mydf.rename( columns = { 'pChEMBL Value': 'PCHEMBL_VALUE', 'Smiles': 'CANONICAL_SMILES' }, inplace=True, errors="raise" )
        return

    def purge_datafile(self) :
        self.mydf = self.mydf[ self.mydf.PCHEMBL_VALUE > 0 ]
        self.mydf.sort_values( by='PCHEMBL_VALUE', ascending=False, inplace=True )
        self.mydf.drop_duplicates( subset=['CANONICAL_SMILES'], keep='first', inplace=True )
        return

    def make_fingerprint( self, nBits=1024 ) :
        self.mydf[ 'fingerprint' ] = self.mydf.apply( lambda row: GenerateMoleculeFingerprint(row,nBits), axis=1 )
        return

    def make_activityclass( self, activity_criteria=6.0 ) :
        self.mydf[ 'activity_class' ] = self.mydf.apply( lambda row: 1 if row.PCHEMBL_VALUE > activity_criteria else 0, axis=1 )
        self.mydf = self.mydf.drop( [ 'PCHEMBL_VALUE', 'CANONICAL_SMILES' ], axis=1 )
        return

    def splitTrainTestSets( self, frac_test=0.3 ) :
        return train_test_split( self.mydf, test_size=frac_test )
    


class DNN( models.Sequential):
	def __init__( self, Nin, Nh_1, Nout ) :
		super().__init__()
		self.add( layers.Dense( Nh_1[0], activation = 'relu', input_shape = (Nin,), name='Hidden-1' ) )
		self.add( layers.Dense( Nh_1[1], activation = 'relu', input_shape = (Nin,), name='Hidden-2' ) )
		self.add( layers.Dense( Nout, activation = 'sigmoid' ) )
		self.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )



def gettraindata( trainset ) :
    x  = np.asarray( trainset['fingerprint'].to_list() ).astype( np.float32 )
    y  = np.asarray( trainset['activity_class'].to_numpy() ).astype( np.int )
    y = to_categorical(y,2)
    return x, y



def func_train( trainset ) :
    x_train, y_train  = gettraindata( trainset )
    # print( '\n\n===> ', type(x_train), x_train.ndim, x_train.size, x_train.shape, y_train.shape, "\n\n" )
    
    number_of_class = 2
    Nin = x_train.shape[1]
    Nh_1 = [ Nin, Nin, 2 ]
    print( '\n===> network dimension = ', Nh_1 )

    model = DNN( Nin, Nh_1, number_of_class )
    history = model.fit( x_train, y_train, epochs=100, batch_size=128, validation_split=0.3, verbose=0)
    # print( '\n===>Training results : ', model, history )
    model.summary()

    return model, history



def print_result( y1, y2, title ) :
    print( "\n\n\n===> ", len(y1), " : ", len(y2), " : ", len(title) )
    for i in range(len(y1)) :
        try :
            print( y1[i], ", ", y2[i], ", ", title[i] )
        except Exception:
            pass
    print( "\n===> ", len(y1), " : ", len(y2), " : ", len(title) )
    return



def func_predict( model, testset, verbose=False ) :
    x_test, y_test  = gettraindata( testset )
    loss, accuracy = model.evaluate( x_test, y_test, batch_size=128 )
    print( '\n===>Test Loss, accuracy, f1_score, precision, recall ->', loss, accuracy )
    y_pred = model.predict( x_test )
    y_pred = np.argmax( y_pred, axis=1 )
    
    if True == verbose :
        np.set_printoptions(threshold=sys.maxsize)
#        print( np.array_repr(y_test).replace('\n', ''), np.array_repr(y_pred).replace('\n', '') )
#        print( "\n".join( "{} {}".format(x,y) for x,y in zip(y_test,y_pred,testset['ID']) ) )
        print_result( np.argmax( y_test, axis=1), y_pred, testset['ID'] )

    confusion = confusion_matrix( np.argmax( y_test, axis=1), y_pred )
    print( confusion )
    return y_pred



def plot_loss( history, fname ) :
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc=0)
    plt.savefig( fname, dpi=300 )



def main_0() :
    nBits = 2048
    activity_criteria = 5.0
    frac_test = 0.3

    datafile = '../data/bacteria/chembl28_bacteria_activity.csv'
    # datafile = '../data/hERG/hERG_ChEMBL27.csv'
    df = prepare_data( datafile, nBits, activity_criteria )
    printNumActivityClass( 'Total', df )

    for iter in range(1) :
        train_set, test_set = splitTrainTestSets( df, frac_test )
        printNumActivityClass( 'Training', train_set )
        printNumActivityClass( 'Test', test_set )

        model, history = func_train( train_set )
        plot_loss( history, 'classification-loss-' + str(iter) + '.png' )

        y_pred = func_predict( model, test_set )

    return



def main_1() :
    nBits = 2048
    activity_criteria = 5.0
    frac_test = 0.3

    datafile = '../data/bacteria/chembl28_bacteria_activity.csv'
    # datafile = '../data/hERG/hERG_ChEMBL27.csv'
    df = MyDataFrame( datafile, nBits, activity_criteria )
    printNumActivityClass( 'Total', df.mydf )

    for iter in range(1024) :
        train_set, test_set = df.splitTrainTestSets( frac_test )
        printNumActivityClass( 'Training', train_set )
        printNumActivityClass( 'Test', test_set )

        model, history = func_train( train_set )
        plot_loss( history, 'classification-loss-' + str(iter) + '.png' )

        y_pred = func_predict( model, test_set )

    return



def prepare_testset( testfilename, title, nBits=2048 ) :
    testdf = pd.read_csv( testfilename, names=[title, 'ID'], delimiter=' ', header=None, nrows=9999999 )
    testdf[ 'fingerprint' ] = testdf.apply( lambda row: GenerateMoleculeFingerprint(row,nBits), axis=1 )
    testdf = testdf.dropna()
    testdf[ 'activity_class' ] = 0
    return testdf



def main() :
    nBits = 2048
    activity_criteria = 6.5
    frac_test = 0.3

    datafile = '../data/bacteria/chembl28_bacteria_activity.csv'
    df = MyDataFrame( datafile, nBits, activity_criteria )
    printNumActivityClass( 'Total', df.mydf )

    for iter in range(1) :
        train_set, test_set = df.splitTrainTestSets( frac_test )
        printNumActivityClass( 'Training', train_set )
        printNumActivityClass( 'Test', test_set )

        model, history = func_train( train_set )
        plot_loss( history, 'classification-loss-' + str(iter) + '.png' )

#        y_pred = func_predict( model, train_set )
#        y_pred = func_predict( model, test_set )

        testdf = prepare_testset( '/share/databases/KCB/00_PI_DB/046_LEEJY/046_KCBDB_KRICT_LeeJooYoun_20211111-00.smiles', 'CANONICAL_SMILES', nBits )
#        testdf = prepare_testset( './DltA_Patent_US2010_0130489.smiles', 'CANONICAL_SMILES', nBits )
        y_predtest = func_predict( model, testdf, True )

    return




if __name__ == '__main__' :
    main()


