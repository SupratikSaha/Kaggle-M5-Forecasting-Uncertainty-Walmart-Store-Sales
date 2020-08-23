""" Code file to create keras models to be utilized in model training """

import gc
from keras import backend as k
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.layers import Dense, Input, Embedding, Dropout, concatenate, Flatten, BatchNormalization
from keras.models import Model


def keras_model(num_dense_features: int, lr: float = 0.002, embedding: int = None) -> Model:
    """ Creates a keras Model using event name embeddings
    Args:
        num_dense_features: Number of numerical columns in input data
        lr: Learning rate of the model
        embedding: Choice of embedding shape used in the model
    Returns:
        Keras model that uses event name embeddings
    """
    k.clear_session()
    gc.collect()

    # Dense input
    dense_input = Input(shape=(num_dense_features,), name='dense1')

    # Embedding input
    wday_input = Input(shape=(1,), name='wday')
    event_name_1_input = Input(shape=(1,), name='event_name_1')
    event_type_1_input = Input(shape=(1,), name='event_type_1')
    event_name_2_input = Input(shape=(1,), name='event_name_2')
    event_type_2_input = Input(shape=(1,), name='event_type_2')
    dept_id_input = Input(shape=(1,), name='dept_id')
    store_id_input = Input(shape=(1,), name='store_id')
    cat_id_input = Input(shape=(1,), name='cat_id')
    state_id_input = Input(shape=(1,), name='state_id')

    wday_emb = Flatten()(Embedding(7, 2)(wday_input))
    event_name_1_emb = Flatten()(Embedding(31, embedding)(event_name_1_input))
    event_type_1_emb = Flatten()(Embedding(5, embedding)(event_type_1_input))
    event_name_2_emb = Flatten()(Embedding(5, embedding)(event_name_2_input))
    event_type_2_emb = Flatten()(Embedding(5, embedding)(event_type_2_input))

    dept_id_emb = Flatten()(Embedding(7, 2)(dept_id_input))
    store_id_emb = Flatten()(Embedding(10, 3)(store_id_input))
    cat_id_emb = Flatten()(Embedding(3, 2)(cat_id_input))
    state_id_emb = Flatten()(Embedding(3, 2)(state_id_input))

    # Combine dense and embedding parts and add dense layers. Exit on linear scale.
    x = concatenate([dense_input,
                     wday_emb,
                     event_name_1_emb, event_type_1_emb,
                     event_name_2_emb, event_type_2_emb,
                     dept_id_emb, store_id_emb,
                     cat_id_emb, state_id_emb])

    x = Dense(256 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256 * 2, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(4 * 2, activation="relu")(x)
    x = BatchNormalization()(x)

    outputs = Dense(1, activation="linear", name='output')(x)

    inputs = [dense_input,
              wday_input,
              event_name_1_input, event_type_1_input,
              event_name_2_input, event_type_2_input,
              dept_id_input, store_id_input,
              cat_id_input, state_id_input]

    # Connect input and output
    model = Model(inputs, outputs)

    model.compile(loss=mean_squared_error,
                  metrics=["mse"],
                  optimizer=Adam(lr=lr))

    return model


def keras_model_no_en1_en2(num_dense_features: int, lr: float = 0.002,
                           embedding: int = None) -> Model:
    """ Creates a keras Model not using event name embeddings
    Args:
        num_dense_features: Number of numerical columns in input data
        lr: Learning rate of the model
        embedding: Choice of embedding shape used in the model
    Returns:
        Keras model that does not use event name embeddings
    """
    k.clear_session()
    gc.collect()

    # Dense input
    dense_input = Input(shape=(num_dense_features,), name='dense1')

    # Embedding input
    wday_input = Input(shape=(1,), name='wday')
    event_type_1_input = Input(shape=(1,), name='event_type_1')
    event_type_2_input = Input(shape=(1,), name='event_type_2')
    dept_id_input = Input(shape=(1,), name='dept_id')
    store_id_input = Input(shape=(1,), name='store_id')
    cat_id_input = Input(shape=(1,), name='cat_id')
    state_id_input = Input(shape=(1,), name='state_id')

    wday_emb = Flatten()(Embedding(7, 2)(wday_input))
    event_type_1_emb = Flatten()(Embedding(5, embedding)(event_type_1_input))
    event_type_2_emb = Flatten()(Embedding(5, embedding)(event_type_2_input))

    dept_id_emb = Flatten()(Embedding(7, 2)(dept_id_input))
    store_id_emb = Flatten()(Embedding(10, 3)(store_id_input))
    cat_id_emb = Flatten()(Embedding(3, 2)(cat_id_input))
    state_id_emb = Flatten()(Embedding(3, 2)(state_id_input))

    # Combine dense and embedding parts and add dense layers. Exit on linear scale.
    x = concatenate([dense_input,
                     wday_emb,
                     event_type_1_emb,
                     event_type_2_emb,
                     dept_id_emb, store_id_emb,
                     cat_id_emb, state_id_emb])

    x = Dense(256 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256 * 2, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16 * 2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Dense(4 * 2, activation="relu")(x)
    x = BatchNormalization()(x)

    outputs = Dense(1, activation="linear", name='output')(x)

    inputs = [dense_input,
              wday_input,
              event_type_1_input,
              event_type_2_input,
              dept_id_input, store_id_input,
              cat_id_input, state_id_input]

    # Connect input and output
    model = Model(inputs, outputs)

    model.compile(loss=mean_squared_error,
                  metrics=["mse"],
                  optimizer=Adam(lr=lr))

    return model
