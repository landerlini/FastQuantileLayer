from __future__ import print_function 
from __future__ import absolute_import 

import tensorflow as tf
import numpy as np 

from scipy.special import erfinv as ierf 

try: 
  from FixedBinInterpolator import FixedBinInterpolator 
except: 
  from .FixedBinInterpolator import FixedBinInterpolator 


class FastQuantileLayer ( tf.keras.layers.Layer ) :
  """
    Creates a keras layer to emulate the behaviour of 
    scikit-learn QuantileTransformer.
  """

  def __init__ (self, 
      n_quantiles = 100, 
      n_sample_invert = 5000, 
      output_distribution='uniform', 
      default_to_inverse = False, 
      numpy_dtype = np.float32, 
      **kwargs
    ):
    """
      n_quantiles : int (default: 100)
        Number of quantiles to be computed. It corresponds to 
        the number of landmarks used to discretize the cumulative 
        density function.

      n_sample_invert : int (default: 5000)
        Number of points used to sample the inverted transform.
        Larger values will result in slower evaluation but more 
        accurate function inversion. 

      output_distribution : string (default: 'uniform')
        Marginal distribution for the transformed data. 
        The choices are 'uniform' (default) or 'normal'.
        The normal distribution is truncated. 

      dtype : numpy data type (default: np.float32)
        Data type of the expected input 

      default_to_inverse : bool
        If default_to_inverse is True, and inverse is explicitely specified
        when applying the layer. 
    """

    self._Nbins             = n_quantiles
    self._Ninv              = n_sample_invert
    self._outDist           = output_distribution
    self.default_to_inverse = default_to_inverse
    self.numpy_dtype        = numpy_dtype 

    self.fwdTransforms_ = [] 
    self.bwdTransforms_ = [] 
    
    tf.keras.layers.Layer.__init__ ( self, kwargs ) 

  def fit ( self, X, y = None ): 
    """
      Creates the tensorflow interpolator used to transform the 
      distribution to either a uniform or normal distribution.  
    """
    rank = len(X.shape) 
    if rank == 1:   # single variable  
      self._fit_column ( X, y ) 

    elif rank == 2: # dataset  
      for iCol in range ( X.shape[1] ): 
        self._fit_column ( X[:,iCol], y ) 
    else:
      raise ValueError ("Expected a numpy array of rank 1 or 2, got %d"%rank)

    return self 


  def build ( self, input_shape ):
    tf.keras.layers.Layer.build ( self, input_shape ) 
  

  def _fit_column ( self, X, y=None ):
    """
      Internal. Creates the interpolator for a single variable 
    """

    ## Fill the histogram 
    hist, edges = np.histogram ( X, bins = self._Nbins ) 
    ## Creates the underflow bin 
    hist = np.concatenate ( [ [0.], hist], axis = 0 ) 

    ## Computes the cumulative distribution 
    y = np.cumsum ( hist , dtype=self.numpy_dtype) 
    y /= y [-1] 

    ## Transform the cumulative distribution to output a 
    ## normal distribution if requested
    if self._outDist == 'normal':
      y = ierf ( np.clip(2.*y - 1.,-0.99, 0.99))  * np.sqrt(2) 
#      y *= ierf ( np.linspace ( -1+1e-10, 1-1e-10, len ( y ) ) )

    ## Prepares the forward transform
    self.fwdTransforms_ . append (
      FixedBinInterpolator ( edges[0], edges[-1], y )
    )

    ## Prepares the backward transform
    y_axis = np.linspace ( y[0], y[-1], self._Ninv )

    self.bwdTransforms_ . append (
        FixedBinInterpolator ( y_axis[0], y_axis[-1], 
        np.interp ( y_axis, y, edges ).astype(self.numpy_dtype), 
      )
    )


  def transform ( self, X, inverse = False ) : 
    """
      Apply the tensorflow graph 
    """
    transf = self.bwdTransforms_ if inverse else self.fwdTransforms_

    if not len(transf): 
      raise RuntimeError ( "QuantileTransformTF was not initialized. Run qtf.fit(numpy_dataset)." ) 

    rank = len(X.shape) 
    if rank == 1:
      return transf[0].apply ( X[:,i] )  
    elif rank == 2:
      return tf.stack ( 
        [ transf[i].apply ( X[:,i] ) for i in range(X.shape[1]) ], 
        axis=1
      )

  def call ( self, X ):
    """
      Service function to call transform 
    """
    return self.transform ( X, self.default_to_inverse ) 

  
  def get_inverse ( self ):
    """
      Return a clone of this layer. 
    """
    new_layer = self.from_config ( self . get_config() ) 
    new_layer . default_to_inverse = not new_layer . default_to_inverse
    return new_layer 

  
  def get_config ( self ):
    """
      Returns the configuration dictionary.
    """
    cfg = tf.keras.layers.Layer.get_config ( self )
    cfg . update ( dict(
        _Nbins             = self._Nbins                 ,   
        _Ninv              = self._Ninv                  , 
        _outDist           = self._outDist               , 
        numpy_dtype        = str(self.numpy_dtype)       , 
        default_to_inverse = self.default_to_inverse     ,   
        direct_transforms  = [
          transform.get_config() for transform in self.fwdTransforms_
        ],
        inverse_transforms = [
          transform.get_config() for transform in self.bwdTransforms_
        ],
    ))
    return cfg 

  
  @classmethod
  def from_config ( cls, cfg ):
    """
      Returns the configuration dictionary.
    """
    newLayer = FastQuantileLayer() 
    newLayer._Nbins               = cfg [ '_Nbins' ] 
    newLayer._Ninv                = cfg [ '_Ninv' ] 
    newLayer.numpy_dtype          = cfg [ 'numpy_dtype'] 
    newLayer.default_to_inverse   = cfg [ 'default_to_inverse' ] 
    newLayer.fwdTransforms_       = [] 
    newLayer.bwdTransforms_       = [] 
    
    for transform in cfg [ 'direct_transforms' ]:
      newLayer.fwdTransforms_ . append ( 
        FixedBinInterpolator ( transform['x_min'], transform['x_max'], 
          np.array(transform['y_values'], dtype=transform ['dtype'] ))
      )

    for transform in cfg [ 'inverse_transforms' ]:
      newLayer.bwdTransforms_ . append ( 
        FixedBinInterpolator ( transform['x_min'], transform['x_max'], 
          np.array(transform['y_values'], dtype=transform ['dtype'] ))
      )
    return newLayer





if __name__ == '__main__':
  dataset = np.c_[
    np.random.uniform ( 0., 1., 1000) , 
    np.random.uniform ( 2., 3., 1000) , 
  ]

  transformer = FastQuantileLayer (dataset.shape[1], output_distribution='normal')
  transformer . fit ( dataset ) 

  transformer . from_config ( transformer.get_config() ) 

  test_dataset = tf.constant(np.c_[
    np.random.uniform ( 0., 1., 10000) , 
    np.random.uniform ( 2., 3., 10000) , 
  ], dtype = tf.float32)

  t = transformer . transform ( test_dataset ) 

  bkwd = transformer . transform ( t, inverse=True ) 

  with tf.Session() as session: 
    print ("###### Original dataset ####### " ) 
    print ("Mean: ", np.mean ( test_dataset.eval() , axis= 0) ) 
    print ("Std: ",  np.std  ( test_dataset.eval() , axis= 0) ) 
    print () 
    print ("###### Forward transform ####### " ) 
    print ("Mean:", np.mean((t.eval()), axis= 0))
    print ("Std: ", np.std ((t.eval()), axis= 0))
    print () 
    print ("###### Backward transform ####### " ) 
    print ("Mean: ", np.mean ( bkwd.eval() , axis= 0) ) 
    print ("Std: ",  np.std  ( bkwd.eval() , axis= 0) ) 
    print () 
    print ("Average squared error: ", np.sqrt(np.mean ( np.square ( test_dataset.eval() - bkwd.eval() ) ))) 
    print ("Max.    squared error: ", np.sqrt(np.max  ( np.square ( test_dataset.eval() - bkwd.eval() ) ))) 
    cmpr = np.c_[test_dataset.eval(), t.eval(),  bkwd.eval()] 
    error = np.abs(cmpr[:,0]-cmpr[:,1]) 

    print ( "Largest errors: " ) 
    print (cmpr [np.argsort(-error)][:10] ) 
    

    
  
  
