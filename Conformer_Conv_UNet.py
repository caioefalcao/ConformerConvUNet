"""
Adaptação do modulo convulucional Conformer como extrator de características da arquitetura padrão UNET usando TensorFlow .

@ Author: Caio Falcão caioefalcao@nca.ufma.br

@ Date created: Abr 20, 2024

@ Date created: Abr 26, 2024

"""
import tensorflow as tf
from keras.utils import plot_model  
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, UpSampling2D,ReLU,LeakyReLU,Add,GlobalAveragePooling2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model, Sequential

#from keras_flops import get_flops


import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'



########################################################
############### UNet Architecture ######################
########################################################

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False,):
    x = layers.Conv2D(num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def CONFORMER_CONV_UNET(image_size, num_classes,activation):
    
    model_input = keras.Input(shape=(image_size, image_size, 3))
    
    ##################### UNet Encoder Call ####################
    skip_list = []
    skip_list, x = UNet_Enconder(model_input)   
    
    
    ########## Decoder Block UNet Shape ###########
    n_filtro = 48
    #Level 4 - #Upsample 2x2 -  #Concate Skip4 + Up1 - Conv3x3 - Conv3x3
    up1 = layers.UpSampling2D(size=2,interpolation="bilinear")(x)
    x = layers.Concatenate(axis=-1)([skip_list[3], up1])
    x = convolution_block( x, num_filters=n_filtro*8, kernel_size=3)
    out_lvl4 = convolution_block( x, num_filters=n_filtro*8, kernel_size=3)
    
    #Level 3 - #Upsample 2x2 -  #Concate Skip3 + Up2 - Conv3x3 - Conv3x3
    up2 = layers.UpSampling2D(size=2,interpolation="bilinear")(out_lvl4)
    x = layers.Concatenate(axis=-1)([skip_list[2], up2])
    x = convolution_block( x, num_filters=n_filtro*4, kernel_size=3)
    out_lvl3 = convolution_block( x, num_filters=n_filtro*4, kernel_size=3)
    
    #Level 2 - #Upsample 2x2 -  #Concate Skip2 + Up3 - Conv3x3 - Conv3x3
    up3 = layers.UpSampling2D(size=2,interpolation="bilinear")(out_lvl3)
    x = layers.Concatenate(axis=-1)([skip_list[1], up3])
    x = convolution_block( x, num_filters=n_filtro*2, kernel_size=3)
    out_lvl2 = convolution_block( x, num_filters=n_filtro*2, kernel_size=3)
   
    #Level 1 - #Upsample 2x2 -  #Concate Skip1 + Up4 - Conv3x3 - Conv3x3
    up4 = layers.UpSampling2D(size=2,interpolation="bilinear")(out_lvl2)
    x = layers.Concatenate(axis=-1)([skip_list[0], up4])
    x = convolution_block( x, num_filters=n_filtro, kernel_size=3)
    out_lvl1 = convolution_block( x, num_filters=n_filtro, kernel_size=3)
   
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same",activation=activation)(out_lvl1)
    return keras.Model(inputs=model_input, outputs=model_output)



################################################################
############### UNET Architecture Encoder ######################
################################################################
def bottleneck_block(entered_input, filters=64):
    # Taking first input and implementing the first conv block
    conv1 = Conv2D(filters, kernel_size = (3,3), padding = "same")(entered_input)
    batch_norm1 = BatchNormalization()(conv1)
    act1 = ReLU()(batch_norm1)
    
    # Taking first input and implementing the second conv block
    conv2 = Conv2D(filters, kernel_size = (3,3), padding = "same")(act1)
    batch_norm2 = BatchNormalization()(conv2)
    act2 = ReLU()(batch_norm2)
    
    return act2

# def UNet_Block(entered_input,filters=64):
#     kernel = (3,3)
#     conv = Conv2D(filters, 
#                   kernel_size = kernel,  # filter size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(entered_input)
#     conv = Conv2D(filters, 
#                   kernel_size = kernel,  # filter size
#                   activation='relu',
#                   padding='same',
#                   kernel_initializer='HeNormal')(conv)
  
#     conv = BatchNormalization()(conv, training=False)
#     next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)   
#     skip_connection = conv 

#     return skip_connection,next_layer


def UNet_Enconder(input):
    # Take the image size and shape
    input1 = input
    n_filtro = 48

    # Construct the encoder blocks 
    skip1, encoder_1 = conformerConvModule(input1, n_filtro)
    skip2, encoder_2 = conformerConvModule(encoder_1,  n_filtro*2)
    skip3, encoder_3 = conformerConvModule(encoder_2, n_filtro*4)
    skip4, encoder_4 = conformerConvModule(encoder_3, n_filtro*8)
        
    # Preparing the next block
    conv_block = bottleneck_block(encoder_4,  n_filtro*16)
    
    return [skip1,skip2,skip3,skip4],conv_block
########################################################
############### ASPP Architecture ############
########################################################


########################################################
############### Conformer Conv Architecture ############
########################################################
# GLU Activation
class GatedLinearUnit(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.linear = tf.keras.layers.Dense(units)
        self.sigmoid = tf.keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)
#BatchNorm
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, causal, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.causal = causal

    def call(self, inputs):
        if not self.causal:
            return tf.keras.layers.BatchNormalization(axis=-1)(inputs)
        return tf.identity(inputs)
#Swish Activation
class Swish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * tf.sigmoid(inputs)
#DepthwiseLayer
class DepthwiseLayer(tf.keras.layers.Layer):
    def __init__(self, chan_in, chan_out, kernel_size, padding, **kwargs):
        super(DepthwiseLayer, self).__init__(**kwargs)
        self.padding = padding
        self.chan_in = chan_in
        self.conv = tf.keras.layers.Conv1D(chan_out, kernel_size, groups=chan_in)

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1])
        padded = tf.zeros(
            [self.chan_in * self.chan_in] - tf.shape(inputs), dtype=inputs.dtype
        )
        inputs = tf.concat([inputs, padded], 0)
        inputs = tf.reshape(inputs, [-1, self.chan_in, self.chan_in])

        return self.conv(inputs)

#Conformer Conv Module Architecture
def conformerConvModule(model_input,filters=64, kernel_size=3):
    dim = filters
    # Create skip connection Residual
    x_skip_res = model_input
    #dim = model_input.shape[-2]
    expansion_factor = 2
    inner_dim = dim * expansion_factor
    dropout=0.0

    ln = tf.keras.layers.LayerNormalization(axis=-1)(model_input)
    pointConv = tf.keras.layers.Conv1D(filters=inner_dim * 2, kernel_size=1)(ln)
    act_glu = GatedLinearUnit(units=inner_dim * 2)(pointConv)
    convDeth = tf.keras.layers.Conv1D(filters=inner_dim,                         # 1D Depthwise Conv
                                   kernel_size=kernel_size,
                                   padding='same',
                                   groups=inner_dim)(act_glu)
    batch_norm1 = BatchNormalization()(convDeth)
    act_swish = Swish()(batch_norm1)
    conv1 = tf.keras.layers.Conv1D(filters=dim, kernel_size=1)(act_swish)
    drop = tf.keras.layers.Dropout(dropout)(conv1)
   
    #merged = keras.layers.concatenate([model_input,drop], axis=-1)
    x_skip_res = tf.keras.layers.Conv2D(filters=dim, kernel_size=(1,1), strides=(1,1))(x_skip_res)
    add_out = Add()([x_skip_res,drop])

    #Adapted for Unet Encoder
    next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(add_out)   
    skip_connection = add_out 
    return skip_connection,next_layer

def mytest():
    ########################################################
    ################### Define Model #######################
    ########################################################
    NUM_CLASSES = 3
    IMAGE_SIZE = 256
    
    
    model = CONFORMER_CONV_UNET(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, activation="softmax")
    model.summary()
    patha="G:\\Meu Drive\\!Doutorado_UFMA-UFPI\\!Codes\\PPM\\Revista\\Revista\\Customizando_Bloco_PPM\\1 - Conformer Conv_UNet copy\\"
    plot_model(model, to_file= patha + "model_plot_UNet_ConformerConv2.png", show_shapes=True, show_layer_names=True)
    
    
    model.save(patha+"keras_model.h5")

if __name__ == '__main__':
    mytest()

