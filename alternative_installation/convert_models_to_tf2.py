import os, sys
import h5py
from deepEMhancer.config import DEFAULT_MODEL_DIR

if len(sys.argv)==2:
  dirname = os.path.expanduser(sys.argv[1])
else:
  dirname = DEFAULT_MODEL_DIR

a="N"
a= input("WARNING, you are going to override all the models contained in %s to make them compatible with tensorflow 2."
      " This operation cannot be undone and will break tensorflow version 1 compatibility (you may want to create a backup "
      " of the original files). Do you want to continue (y/N)"%dirname).strip().lower()
if a == "y" or a == "yes":

  for fname in os.listdir(dirname):
    if fname.endswith(".txt"): continue
    fname = os.path.join( dirname, fname)
    print("Modifying %s"%fname)
    with h5py.File(fname,'r+') as h5File:
      #print( h5File['code']['custom_objects'][0] )
      h5File['code']['custom_objects'][0] ='''
    ##Modified from keras_contrib repository
    import tensorflow as tf
    from tensorflow.keras.layers import Layer, InputSpec

    class GroupNormalization(Layer):
        """Group normalization layer.

        Group Normalization divides the channels into groups and computes
        within each group
        the mean and variance for normalization.
        Group Normalization's computation is independent
         of batch sizes, and its accuracy is stable in a wide range of batch sizes.

        Relation to Layer Normalization:
        If the number of groups is set to 1, then this operation becomes identical to
        Layer Normalization.

        Relation to Instance Normalization:
        If the number of groups is set to the
        input dimension (number of groups is equal
        to number of channels), then this operation becomes
        identical to Instance Normalization.

        # Arguments
            groups: Integer, the number of groups for Group Normalization.
                Can be in the range [1, N] where N is the input dimension.
                The input dimension must be divisible by the number of groups.
            axis: Integer, the axis that should be normalized
                (typically the features axis).
                For instance, after a `Conv2D` layer with
                `data_format="channels_first"`,
                set `axis=1` in `BatchNormalization`.
            epsilon: Small float added to variance to avoid dividing by zero.
            center: If True, add offset of `beta` to normalized tensor.
                If False, `beta` is ignored.
            scale: If True, multiply by `gamma`.
                If False, `gamma` is not used.
                When the next layer is linear (also e.g. `nn.relu`),
                this can be disabled since the scaling
                will be done by the next layer.
            beta_initializer: Initializer for the beta weight.
            gamma_initializer: Initializer for the gamma weight.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_constraint: Optional constraint for the beta weight.
            gamma_constraint: Optional constraint for the gamma weight.

        # Input shape
            Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.

        # Output shape
            Same shape as input.

        # References
            - [Group Normalization](https://arxiv.org/abs/1803.08494)
        """

        def __init__(self,
                     groups=32,
                     axis=-1,
                     epsilon=1e-5,
                     center=True,
                     scale=True,
                     beta_initializer='zeros',
                     gamma_initializer='ones',
                     beta_regularizer=None,
                     gamma_regularizer=None,
                     beta_constraint=None,
                     gamma_constraint=None,
                     **kwargs):
            super(GroupNormalization, self).__init__(**kwargs)
            self.supports_masking = True
            self.groups = groups
            self.axis = axis
            self.epsilon = epsilon
            self.center = center
            self.scale = scale
            self.beta_initializer = tf.keras.initializers.get(beta_initializer)
            self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
            self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
            self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
            self.beta_constraint = tf.keras.constraints.get(beta_constraint)
            self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

        def build(self, input_shape):
            dim = input_shape[self.axis]

            if dim is None:
                raise ValueError('Axis ' + str(self.axis) + ' of '
                                 'input tensor should have a defined dimension '
                                 'but the layer received an input with shape ' +
                                 str(input_shape) + '.')

            if dim < self.groups:
                raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                                 'more than the number of channels (' +
                                 str(dim) + ').')

            if dim % self.groups != 0:
                raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                                 'multiple of the number of channels (' +
                                 str(dim) + ').')

            self.input_spec = InputSpec(ndim=len(input_shape),
                                        axes={self.axis: dim})
            shape = (dim,)

            if self.scale:
                self.gamma = self.add_weight(shape=shape,
                                             name='gamma',
                                             initializer=self.gamma_initializer,
                                             regularizer=self.gamma_regularizer,
                                             constraint=self.gamma_constraint)
            else:
                self.gamma = None
            if self.center:
                self.beta = self.add_weight(shape=shape,
                                            name='beta',
                                            initializer=self.beta_initializer,
                                            regularizer=self.beta_regularizer,
                                            constraint=self.beta_constraint)
            else:
                self.beta = None
            self.built = True

        def call(self, inputs, **kwargs):
            input_shape = tf.keras.backend.int_shape(inputs)
            tensor_input_shape = tf.keras.backend.shape(inputs)

            # Prepare broadcasting shape.
            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(1, self.groups)

            reshape_group_shape = tf.keras.backend.shape(inputs)
            group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
            group_axes[self.axis] = input_shape[self.axis] // self.groups
            group_axes.insert(1, self.groups)

            # reshape inputs to new group shape
            group_shape = [group_axes[0], self.groups] + group_axes[2:]
            group_shape = tf.keras.backend.stack(group_shape)
            inputs = tf.keras.backend.reshape(inputs, group_shape)

            group_reduction_axes = list(range(len(group_axes)))
            mean, variance = tf.nn.moments(inputs, group_reduction_axes[2:],
                                        keepdims=True)
            inputs = (inputs - mean) / (tf.keras.backend.sqrt(variance + self.epsilon))

            # prepare broadcast shape
            inputs = tf.keras.backend.reshape(inputs, group_shape)

            outputs = inputs

            # In this case we must explicitly broadcast all parameters.
            if self.scale:
                broadcast_gamma = tf.keras.backend.reshape(self.gamma, broadcast_shape)
                outputs = outputs * broadcast_gamma

            if self.center:
                broadcast_beta = tf.keras.backend.reshape(self.beta, broadcast_shape)
                outputs = outputs + broadcast_beta

            # finally we reshape the output back to the input shape
            outputs = tf.keras.backend.reshape(outputs, tensor_input_shape)

            return outputs

        def get_config(self):
            config = {
                'groups': self.groups,
                'axis': self.axis,
                'epsilon': self.epsilon,
                'center': self.center,
                'scale': self.scale,
                'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
                'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
                'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
                'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
                'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
                'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint)
            }
            base_config = super(GroupNormalization, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        def compute_output_shape(self, input_shape):
            return input_shape


    custom_objects= {"GroupNormalization": GroupNormalization}
    '''

else:
  print("Cancelling... No modifications were made")
