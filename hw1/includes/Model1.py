from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
from keras import metrics
from keras import regularizers

class Model1:
    def create( self, inputShape, learningRate ):
        
        model_input = layers.Input( shape = inputShape )
        
        x = layers.Dense( 32, activation = activations.relu )(model_input)
        x = layers.Dense( 64, activation = activations.relu )(x)
        x = layers.Dense( 128, activation = activations.relu )(x)
        
        o1 = layers.Dense(1, name = 'output1' )(x)
        o2 = layers.Dense(1, name = 'output2' )(x)
        o3 = layers.Dense(1, name = 'output3' )(x)
        
        model = models.Model( model_input, [ o1, o2, o3 ] )
        model.summary()
        model.compile( 
            optimizer = optimizers.rmsprop( lr = learningRate ), 
            loss = losses.mse,
            metrics = [ metrics.mae ] 
        )

        model.name = "32-64-128 multi task"
        return model