import tensorflow as tf
# import prettytensor as pt
import numpy as np
import utils

class FullyConnected( object ):

	def __init__( 	self,
					dim_output,
					hidden_layers = [500],
					nonlinearity = tf.nn.softplus,
					l2loss = 0.0,
					scope = 'FullyConnected'	):

		self.dim_output = dim_output
		self.hidden_layers = hidden_layers
		self.nonlinearity = nonlinearity
		self.l2loss = l2loss
		self.scope = scope

	def output( self, inputs, reuse = False ):

		for i, layer in enumerate(self.hidden_layers):
			inputs = tf.contrib.layers.fully_connected(inputs, num_outputs=layer,
				activation_fn=self.nonlinearity,
				reuse=reuse,
				scope='%s/%d' % (self.scope, i),
				# weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2loss)
				)
		return tf.contrib.layers.fully_connected(inputs, num_outputs=self.dim_output,
			activation_fn=None,
			reuse=reuse,
			scope='%s/%d' % (self.scope, i+1),
			# weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2loss)
			)

