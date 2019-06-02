import numpy as np 
import tensorflow as tf 

class FixedBinInterpolator:
  """
    FixedBinInterpolator creates a graph that evaluates a point-defined
    function y = f(x) through linear interpolation.
    It relies on the assumption that y samples are sorted and obtained 
    from equidistant x values:
      y_i = f ( x_i )  for x_i in np.linspace ( x_min, x_max, len(y_i) ) 
  """

  def __init__ ( self, x_min, x_max, y_values, eps=1e-12 ):
    """ Constructor:
      x_min : float 
        x-value corresponding to the first y in y_values
      x_max : float
        x-value corresponding to the last y in y_values
      y_values : np.ndarray
        y-values sampled at x in np.linspace(x_min, x_max, len(y_i)) 
      eps : float (default: 1e-12)
        a very small positive number compared to the x scale 
    """

    self.y_values = y_values

    self.x_min    = x_min 
    self.x_max    = x_max  
    self.eps      = eps    
    self.dtype    = self.y_values.dtype  

    self.tf_y_values = tf.constant ( y_values , dtype = self.dtype )  
    self.tf_x_min    = tf.constant ( x_min    , dtype = self.dtype )  
    self.tf_x_max    = tf.constant ( x_max    , dtype = self.dtype )  
    self.tf_eps      = tf.constant ( eps      , dtype = self.dtype )   

  def get_config ( self ):
    """ 
      Return a dictionary including the configuration. 
    """
    return dict (
        y_values  = list([float (v) for v in self.y_values]),
        x_min     = float ( self.x_min ),
        x_max     = float ( self.x_max ),
        eps       = float ( self.eps ), 
        dtype     = str(self.dtype), 
      )
    return ret 
      

  def apply ( self, x_values ):
    """ Apply the interpolation graph:
      x_values: 1D tensor
        x-values for which to compute the corresponding y through interpolation

      Returns:
        y_values : 1D tensor of the same length as x_values
    """

    ## Prune the x_values of values larger than the maximum and 
    ## smaller than the minimum for which interpolation would be illdefined 
    ones = tf.ones_like ( x_values ) 

    ## Find the bin 
    N    = int(self.tf_y_values.shape[0])
    R    = (self.tf_x_max - self.tf_x_min)
    x_id = (x_values/R - self.tf_x_min/R) * (N-1) 
    x_0f = tf.floor ( x_id )
    x_0 = tf.cast ( x_0f , tf.int64 ) 

    x_0 = tf.clip_by_value ( x_0, 0, N-2 )
    x_1 = x_0 + 1 


    y_0 = tf.gather ( self.tf_y_values, tf.cast(x_0, tf.int64) ) 
    y_1 = tf.gather ( self.tf_y_values, tf.cast(x_1, tf.int64) ) 

    y = y_0 + tf.clip_by_value(x_id - x_0f,0,1) * (y_1 - y_0)

    return tf.identity ( y, 'interpolated' ) 



if __name__ == '__main__':
  interpolator = FixedBinInterpolator ( 0, 1, [0., 0.5, 1.] ) 

  x = tf.constant ( np.linspace ( 0,1.1, 10000000), dtype=tf.float32) 
  y = interpolator.apply ( x )  

  with tf.Session() as session:
    print ( np.c_ [ x.eval(), y.eval() ] ) 


  

    


    
    
