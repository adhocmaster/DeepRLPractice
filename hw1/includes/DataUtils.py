import numpy as np
import glob
import math
import pickle
import keras
from sklearn.model_selection import train_test_split

class DataUtils:
    
    def __init__( self, inputDir ):
        self.inputDir = inputDir
        self.files = glob.glob( inputDir )
        
        data = {} #this dictionary can have different structures depending on what kind of data we are working on. 
        
        pass
    
    # Returns a dictionary having two keys: trObservations, trActions, tsObservations and tsActions 
    # ( normalized by standard deviation and mean. )
    def processMujocoExpertData( self, file ):
        objects = []
        with (open(file, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
                    
        oShape = self.getObservationShapeWith0Rows( objects )
        aShape = self.getActionShapeWith0Rows( objects )
        observations = np.empty( oShape )
        actions = np.empty( aShape )
        
        for dataset in objects:
            observations = np.vstack( ( observations, dataset['observations'] ) )
            actions = np.vstack( ( actions, dataset['actions'] ) )
            
        #normalize data
        observations = keras.utils.normalize( observations, axis = ( observations.ndim - 1 ) )
        actions = keras.utils.normalize( actions, axis = ( actions.ndim - 1 ) )
            
        return observations, actions
     
    def getObservationShapeWith0Rows( self, objects ):
        
        if not objects:
            raise ValueError( "data objects empty" )
        
        dataset = objects[0]
        oShape = list( dataset['observations'].shape )
        oShape[0] = 0
        return tuple(oShape)
     
    def getActionShapeWith0Rows( self, objects ):
        
        if not objects:
            raise ValueError( "data objects empty" )
        
        dataset = objects[0]
        aShape = list( dataset['actions'].shape )
        aShape[0] = 0
        return tuple(aShape)
    
    
    
    
    