from keras.layers.convolutional import Conv2DTranspose, MaxPooling2D, Conv2D
from keras.layers import Input, concatenate, Dropout

from keras.models import Model

def build_model(input_shape):
    inputs = Input(input_shape)
    DROPOUT = 0.3

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((1, 1))(c1)
    p1 = Dropout(DROPOUT)(p1)


    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((1, 1))(c2)
    p2 = Dropout(DROPOUT)(p2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(DROPOUT)(p3)

    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(32, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(DROPOUT)(p4)

    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(c5)
    p5 = MaxPooling2D((2, 2))(c5)
    p5 = Dropout(DROPOUT)(p5)

    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(p5)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)
    p6 = MaxPooling2D((2, 2))(c6)
    p6 = Dropout(DROPOUT)(p6)

    # c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(p6)
    # c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)
    # p7 = MaxPooling2D((2, 2))(c7)
    # p7 = Dropout(DROPOUT)(p7)

    # c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(p7)
    # c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    # p8 = MaxPooling2D((1, 1))(c8)
    #
    # c9 = Conv2D(128, (3, 3), activation='relu', padding='same')(p8)
    # c9 = Conv2D(128, (3, 3), activation='relu', padding='same')(c9)
    # p9 = MaxPooling2D(pool_size=(1, 1))(c9)
    #
    # c10 = Conv2D(256, (3, 3), activation='relu', padding='same')(p9)
    # c10 = Conv2D(256, (3, 3), activation='relu', padding='same')(c10)
    # p10 = MaxPooling2D(pool_size=(1, 1))(c10)

    # c11 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # c11 = Conv2D(256, (3, 3), activation='relu', padding='same')(c11)
    # p11 = MaxPooling2D(pool_size=(1, 1))(c11)
    #
    # # Now Downsampling the layers
    #
    # u12 = Conv2DTranspose(256, (2, 2), strides=(1, 1), padding='same')(x)
    # u12 = concatenate([u12, c11])
    # c12 = Conv2D(256, (3, 3), activation='relu', padding='same')(u12)
    # c12 = Conv2D(256, (3, 3), activation='relu', padding='same')(c12)

    # u13 = Conv2DTranspose(256, (2, 2), strides=(1, 1), padding='same')(p10)
    # u13 = concatenate([u13, c10])
    # c13 = Conv2D(256, (3, 3), activation='relu', padding='same')(u13)
    # c13 = Conv2D(256, (3, 3), activation='relu', padding='same')(c13)
    #
    # u14 = Conv2DTranspose(128, (2, 2), strides=(1, 1), padding='same')(c13)
    # u14 = concatenate([u14, c9])
    # c14 = Conv2D(128, (3, 3), activation='relu', padding='same')(u14)
    # c14 = Conv2D(128, (3, 3), activation='relu', padding='same')(c14)
    #
    # u15 = Conv2DTranspose(128, (2, 2), strides=(1, 1), padding='same')(c14)
    # u15 = concatenate([u15, c8])
    # c15 = Conv2D(128, (3, 3), activation='relu', padding='same')(u15)
    # c15 = Conv2D(128, (3, 3), activation='relu', padding='same')(c15)

    # u16 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(p7)
    # u16 = concatenate([u16, c7], axis=3)
    # c16 = Conv2D(128, (3, 3), activation='relu', padding='same')(u16)
    # c16 = Conv2D(128, (3, 3), activation='relu', padding='same')(c16)
    # c16 = Dropout(DROPOUT)(c16)

    u17 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(p6)
    u17 = concatenate([u17, c6], axis=3)
    c17 = Conv2D(64, (3, 3), activation='relu', padding='same')(u17)
    c17 = Conv2D(64, (3, 3), activation='relu', padding='same')(c17)
    c17 = Dropout(DROPOUT)(c17
                           )
    u18 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c17)
    u18 = concatenate([u18, c5], axis=3)
    c18 = Conv2D(32, (3, 3), activation='relu', padding='same')(u18)
    c18 = Conv2D(32, (3, 3), activation='relu', padding='same')(c18)
    c18 = Dropout(DROPOUT)(c18)

    u19 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c18)
    u19 = concatenate([u19, c4], axis=3)
    c19 = Conv2D(32, (3, 3), activation='relu', padding='same')(u19)
    c19 = Conv2D(32, (3, 3), activation='relu', padding='same')(c19)
    c19 = Dropout(DROPOUT)(c19)

    u20 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c19)
    u20 = concatenate([u20, c3], axis=3)
    c20 = Conv2D(32, (3, 3), activation='relu', padding='same')(u20)
    c20 = Conv2D(32, (3, 3), activation='relu', padding='same')(c20)
    c20 = Dropout(DROPOUT)(c20)

    u21 = Conv2DTranspose(16, (2, 2), strides=(1, 1), padding='same')(c20)
    u21 = concatenate([u21, c2], axis=3)
    c21 = Conv2D(16, (3, 3), activation='relu', padding='same')(u21)
    c21 = Conv2D(16, (3, 3), activation='relu', padding='same')(c21)
    c21 = Dropout(DROPOUT)(c21)

    u22 = Conv2DTranspose(8, (2, 2), strides=(1, 1), padding='same')(c21)
    u22 = concatenate([u22, c1], axis=3)
    c22 = Conv2D(8, (3, 3), activation='relu', padding='same')(u22)
    c22 = Conv2D(8, (3, 3), activation='relu', padding='same')(c22)
    c22 = Dropout(DROPOUT)(c22)

    outputs = Conv2D(4, (1, 1), activation='sigmoid')(c22)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


