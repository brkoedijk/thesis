# -*- coding: utf-8 -*-
"""
Utility layers
--------------
June 30, 2022
@author: hansbuehler
"""

from .base import Logger, Config, tf, dh_dtype, tf_glorot_value, Int, Float, DIM_DUMMY# NOQA
from collections.abc import Mapping, Sequence # NOQA
import numpy as np
_log = Logger(__file__)

class VariableLayer(tf.keras.layers.Layer):
    """
    A variable layer.
    The variable can be initialized with a specific value, or with the standard Keras glorot initializer.
    """
    
    def __init__(self, init, trainable : bool = True, name : str = None, dtype : tf.DType = dh_dtype ):
        """
        Initializes the variable

        Parameters
        ----------
            init : 
                If a float, a numpy array, or a tensor, then this is the initial value of the variable
                If this is a tuple, a tensorshape, or a numpyshape then this will be the shape of the variable.
            trainable : bool
            name : str
            dtype : dtype
        """        
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )        
        if not isinstance(init, (float, np.ndarray, tf.Tensor)):
            _log.verify( isinstance(init, (tuple, tf.TensorShape)), "'init' must of type float, np.array, tf.Tensor, tuple, or tf.TensorShape. Found type %s", type(init))
            init                 = tf_glorot_value(init)
        self.variable            = tf.Variable( init, trainable=trainable, name=name+"_variable" if not name is None else None, dtype=self.dtype )
        self._available_features = None

    def build( self, shapes : dict ):
        """
        Build the variable layer
        This function ensures 'shapes' contains DIM_DUMMY so it can create returns of sample size
        """
        self._available_features = sorted( [ str(k) for k in shapes if not k == DIM_DUMMY ] )
        dummy_shape = shapes.get(DIM_DUMMY, None)
        _log.verify( not dummy_shape is None, "Every data set must have a member '%s' (see base.DIM_DUMMY) of shape (None,1). Data member not found data: %s", DIM_DUMMY, list(self.available_features) )
        dummy_shape_list = (
            dummy_shape.as_list() if hasattr(dummy_shape, "as_list") else list(dummy_shape)
        )
        _log.verify( len(dummy_shape_list) == 2, "Data set member '%s' (see base.DIM_DUMMY) nust be of shape [None,1], not of shape %s", DIM_DUMMY, dummy_shape_list )
        _log.verify( int(dummy_shape_list[1]) == 1, "Data set member '%s' (see base.DIM_DUMMY) nust be of shape [None,1], not of shape %s", DIM_DUMMY, dummy_shape_list )
        
    def call( self, dummy_data : dict = None, training : bool = False ) -> tf.Tensor:
        """
        Return variable value
        The returned tensor will be of dimension [None,] if self.variable is a float, and otherwise of dimension [None, ...] where '...' refers to the dimension of the variable.        

        The 'dummy_data' dictionary must have an element DIM_DUMMY of dimension (None,).
        """
        dummy = dummy_data[DIM_DUMMY]
        assert len(dummy.shape) == 2, "Internal error: shape %s not (None,)" % str(dummy.shape.as_list())
        x     = tf.zeros_like(dummy[:,0])
        while len(x.shape) <= len(self.variable.shape):
            x = x[:,tf.newaxis,...]
        x = x + self.variable[tf.newaxis,...]
        return x
    
    @property
    def features(self) -> list:
        """ Returns the list of features used """
        return []
    @property
    def available_features(self) -> list:
        """ Returns the list of features avaialble """
        _log.verify( not self._available_features is None, "build() must be called first")
        return self._available_features
    @property
    def nFeatures(self) -> int:
        """ Returns the number of features used """
        return 0
    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The model must have been call()ed once """
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] )

    
class DenseLayer(tf.keras.layers.Layer):
    """
    Core dense Keras layer
    Pretty generic dense layer. Also acts as plain variable if it does not depend on any variables.
    """
    
    def __init__(self, features, nOutput : int, initial_value = None, config : Config = Config(), name : str = None, defaults = Config(), dtype : tf.DType = dh_dtype, use_gru: bool = None, gru_units : int = None ):
        """
        Create a simple dense later with nInput nodes and nOuput nodes.
        
        Parameters
        ----------
            features
                Input features. If None, then the layer will become a simple variable with nOutput nodes.
            nOutput : int
                Number of output nodes
            width : int = 20
            depth : int = 3
            activation : str = "relu"
            name : str, optional
                Name of the layer
            dtype : tf.DType, optional
                dtype
        """
        tf.keras.layers.Layer.__init__(self, name=name, dtype=dtype )
        self.nOutput           = int(nOutput)
        def_width              = defaults("width",20, Int>0, help="Network width.")
        def_activation         = defaults("activation","relu", help="Network activation function")
        def_depth              = defaults("depth", 3, Int>0, help="Network depth")
        def_final_activation   = defaults("final_activation","linear", help="Network activation function for the last layer")
        def_zero_model         = defaults("zero_model", False, bool, "Create a model with zero initial value, but randomized initial gradients")
        self.width             = config("width",def_width, Int>0, help="Network width.")
        self.activation        = config("activation",def_activation, help="Network activation function")
        self.depth             = config("depth", def_depth, Int>0, help="Network depth")
        self.final_activation  = config("final_activation",def_final_activation, help="Network activation function for the last layer")
        self.zero_model        = config("zero_model", def_zero_model, bool, "Create a model with zero initial value, but randomized initial gradients")

        self.use_gru = use_gru if use_gru is not None else config("use_gru", False, bool, "Whether to use GRU")
        self.gru_units = gru_units if gru_units is not None else config("gru_units", 32, Int>0, "Number of GRU units")

        self.features          = sorted( set( features ) ) if not features is None else None
        self.nFeatures         = None
        self.model             = None        
        self.initial_value     = None
        self.available_features= None
        
        if not initial_value is None:
            if isinstance(initial_value, np.ndarray):
                _log.verify( initial_value.shape == (nOutput,), "Internal error: initial value shape %s does not match 'nOutput' of %ld", initial_value.shape, nOutput )
                self.initial_value = initial_value
            else:
                self.initial_value = np.full((nOutput,), initial_value)
                
        _log.verify( self.nOutput > 0, "'nOutput' must be positive; found %ld", self.nOutput )
        config.done()

    def build( self, shapes : dict ):
        """ 
        Keras layer builld() function.
        'shapes' must be a dictionary
        """
        assert self.nFeatures is None and self.model is None, ("build() called twice")
        _log.verify( self.features is None or isinstance(shapes, Mapping), "'shapes' must be a dictionary type if 'features' are specified. Found type %s", type(shapes ))
        
        # collect features
        # features can have different dimensions, so we count the total size of the feature vector
        self.nFeatures = 0
        has_spatial_wind = False
        self.wind_dim = 0
        if not self.features is None:
            if 'wind_info' in self.features and 'wind_info' in shapes:
                wind_shape = shapes['wind_info']
                if wind_shape[1] == 15:
                    has_spatial_wind = True
                    self.wind_dim = wind_shape[1]
            for feature in self.features:
                _log.verify( feature in shapes, "Unknown feature '%s'. Known features are: %s. List of requested features: %s", feature, list(shapes), list(self.features) )
                fs = shapes[feature]
                assert len(fs) == 2, ("Internal error: all features should have been flattend. Found feature '%s' with shape %s" % (feature, fs))
                # if not (has_spatial_wind and feature == 'wind_info'): 
                self.nFeatures += fs[1]
        
                
        self.available_features = sorted( [ str(k) for k in shapes if not k == DIM_DUMMY ] )
        self.has_spatial_wind = has_spatial_wind
        
        # build model
        # simple feedforward model as an example
        if self.nFeatures == 0:
            self.model    = VariableLayer( (self.nOutput,) if self.initial_value is None else self.initial_value, trainable=True, name=self.name+"_variable_layer" if not self.name is None else None, dtype=self.dtype )
        else:
            inp = tf.keras.layers.Input( shape=(self.nFeatures,), dtype=self.dtype )
            x = inp

            if self.use_gru and self.gru_units > 0:
                x = tf.keras.layers.Reshape((1, x.shape[-1]))(x)
                x = tf.keras.layers.GRU(self.gru_units, return_sequences=False, name='gru_layer')(x)

            x = tf.keras.layers.Dense( units=self.width,
                                        activation=self.activation,
                                        use_bias=True )(x)
                                                
            for d in range(self.depth-1):
                x = tf.keras.layers.Dense(units=self.width,
                                            activation=self.activation,
                                            use_bias=True )(x)
            x = tf.keras.layers.Dense(units=self.nOutput,
                                        activation=self.final_activation,
                                        use_bias=True )(x)
                
            self.model         = tf.keras.Model( inputs=inp, outputs=x )

        # if self.has_spatial_wind:
        #     wind_inp = tf.keras.layers.Input(shape=(self.wind_dim,), dtype=self.dtype, name='wind_input')
        #     wind_hidden = tf.keras.layers.Dense(64, activation='relu', name='wind_encoder_1')(wind_inp)
        #     wind_encoded = tf.keras.layers.Dense(8, activation='relu', name='wind_encoder_2')(wind_hidden)
            
        #     if self.nFeatures > 0:
        #         market_inp = tf.keras.layers.Input(shape=(self.nFeatures,), dtype=self.dtype, name='market_input')
        #         combined = tf.keras.layers.Concatenate(name='concat_wind_market')([wind_encoded, market_inp])
        #         inputs = [wind_inp, market_inp]
        #     else:
        #         combined = wind_encoded
        #         inputs = [wind_inp]

        #     x = combined
        # else:
        #     inp = tf.keras.layers.Input(shape=(self.nFeatures,), dtype=self.dtype)
        #     x = inp
        #     inputs = inp

        # if self.use_gru and self.gru_units > 0:
        #     x = tf.keras.layers.Reshape((1, x.shape[-1]))(x)
        #     x = tf.keras.layers.GRU(self.gru_units, return_sequences=False, name='gru_layer')(x)
        
        # x = tf.keras.layers.Dense( units=self.width,
        #                                activation=self.activation,
        #                                use_bias=True )(x)
                                               
        # for d in range(self.depth-1):
        #     x = tf.keras.layers.Dense( units=self.width,
        #                             activation=self.activation,
        #                             use_bias=True )(x)
        # x = tf.keras.layers.Dense(     units=self.nOutput,
        #                             activation=self.final_activation,
        #                             use_bias=True )(x)
        # self.model = tf.keras.Model(inputs=inputs, outputs=x)

        # if self.nFeatures == 0:
        #     """ Create model without inputs, but which is trainable.
        #         Same as creating a plain variable, but wrappong it allows us using
        #         a single self.model
        #     """
        #     self.model    = VariableLayer( (self.nOutput,) if self.initial_value is None else self.initial_value, trainable=True, name=self.name+"_variable_layer" if not self.name is None else None, dtype=self.dtype )
        # else:
        #     """ Simple feed forward network with optional recurrent layer """
            
        #     has_spatial_wind = ('wind_info' in self.features and 'wind_info' in shapes and shapes['wind_info'] == 225)

        #     if has_spatial_wind:
        #         wind_inp = tf.keras.layers.Input(shape=(225,), dtype=self.dtype)
        #         wind_hidden = tf.keras.layers.Dense(64, activation='relu')(wind_inp) # W1, b1, relu
        #         wind_encoded = tf.keras.layers.Dense(8, activation='relu')(wind_hidden)
        #         market_nFeatures = self.nFeatures - 225
        #         market_inp = tf.keras.layers.Input(shape=(market_nFeatures,), dtype=self.dtype)
        #         combined = tf.keras.layers.Concatenate()([wind_encoded, market_inp])

        #         x = combined
        #         x = tf.keras.layers.Dense( units=self.width,
        #                                activation=self.activation,
        #                                use_bias=True )(x)
                                               
        #         for d in range(self.depth-1):
        #             x = tf.keras.layers.Dense( units=self.width,
        #                                     activation=self.activation,
        #                                     use_bias=True )(x)
        #         x = tf.keras.layers.Dense(     units=self.nOutput,
        #                                     activation=self.final_activation,
        #                                     use_bias=True )(x)
        #         self.model = tf.keras.Model(inputs=[wind_inp, market_inp], outputs=x)
        #     else:
        #         inp = tf.keras.layers.Input( shape=(self.nFeatures,), dtype=self.dtype )
        #         x = inp
        #         x = tf.keras.layers.Dense( units=self.width,
        #                                 activation=self.activation,
        #                                 use_bias=True )(x)
                                                
        #         for d in range(self.depth-1):
        #             x = tf.keras.layers.Dense( units=self.width,
        #                                     activation=self.activation,
        #                                     use_bias=True )(x)
        #         x = tf.keras.layers.Dense(     units=self.nOutput,
        #                                     activation=self.final_activation,
        #                                     use_bias=True )(x)
                
                
        #         self.model         = tf.keras.Model( inputs=inp, outputs=x )
            
        if self.zero_model:
            raise NotImplementedError("zero_model")
            """
                cloned = tf.keras.clone_model( self.model, input_tensors=inp )
                assert len(cloned.weights) == len(self.model.weights), "Internal error: cloned model has differnet number of variables?"
                for mvar, cvar in zip( self.model.weights, cloned.weights):
                    cvar.set_weights(mvar.set_weights)
                cloned.trainable = False
                self.model = tf.keras.layers.
            """  
        
    def call( self, data : dict, training : bool = False ) -> tf.Tensor:
        """
        Ask the agent for an action.
    
        Parameters
        ----------
            data : dict
                Contains all available features at this time step.
                This must be a dictionary.
            training : bool, optional
                Whether we are training or not
                
        Returns
        -------
            Tensor with actions. The second dimension of
            the tensor corresponds to self.nInst
    
        """
        _log.verify( self.features is None or isinstance(data, Mapping), "'data' must be a dictionary type. Found type %s", type(data ))
        _log.verify( not self.model is None, "Model has not been buit yet")

        # simple variable --> return as such
        # if self.has_spatial_wind:
        #     # multi-input model for spatial wind field
        #     wind_features = data['wind_info']
        #     other_features = [data[_] for _ in self.features if _ != 'wind_info']

        #     if len(other_features) > 0:
        #         market_features = tf.concat(other_features, axis=1, name='market_features')
        #         return self.model({'wind_input':wind_features, 'market_input':market_features}, training=training)
        #     else:
        #         # Only wind features, no market features
        #         return self.model({'wind_input': wind_features}, training=training)
        # elif self.nFeatures == 0:
        #     self.model(data, training=training)
        # else:
        #     # Regular concatenated features
        #     features = [ data[_] for _ in self.features ]
        #     features = tf.concat( features, axis=1, name = "features" )
    
        #     assert self.nFeatures == features.shape[1], ("Condig error: number of features should match up. Found %ld and %ld" % ( self.nFeatures, features.shape[1] ) )
        #     return self.model( features, training=training )

        if self.nFeatures == 0:
            return self.model(data, training=training)

        features = [data[_] for _ in self.features]
        if len(features) == 1:
            features = features[0]
        else:
            features = tf.concat(features, axis=1, name="features")
 
        assert self.nFeatures == features.shape[1], ("Condig error: number of features should match up. Found %ld and %ld" % ( self.nFeatures, features.shape[1] ) )
        return self.model( features, training=training )
    
    @property
    def num_trainable_weights(self) -> int:
        """ Returns the number of weights. The model must have been call()ed once """
        assert not self.model is None, "build() must be called first"
        weights = self.trainable_weights
        return np.sum( [ np.prod( w.get_shape() ) for w in weights ] )
