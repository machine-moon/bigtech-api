import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Generate some training and testing data
a = np.random.uniform(0, 1, 10000)
b = np.random.uniform(0, 1, 10000)
c = np.sqrt(a**2 + b**2)

# Reshape the data
a = a.reshape(10000, 1)
b = b.reshape(10000, 1)
c = c.reshape(10000, 1)

# Combine a and b into one input array
input_data = np.hstack((a, b))
output_data = c

# Create the neural network
model = Sequential()
model.add(Dense(1, input_dim=2,))
model.add(Dense(1, activation="linear"))

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
model.fit(input_data, output_data, epochs=10, batch_size=10)

# Test the model
a_test = np.random.uniform(0, 1, 10)
b_test = np.random.uniform(0, 1, 10)
c_test = np.sqrt(a_test**2 + b_test**2)

a_test = a_test.reshape(10, 1)
b_test = b_test.reshape(10, 1)

input_test = np.hstack((a_test, b_test))

predictions = model.predict(input_test)

print("Predictions:", predictions)
print("True values:", c_test)


"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image

# Multi-head self-attention mechanism
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)  # Dense layer for query
        self.key_dense = layers.Dense(embed_dim)  # Dense layer for key
        self.value_dense = layers.Dense(embed_dim)  # Dense layer for value
        self.combine_heads = layers.Dense(embed_dim)  # Dense layer to combine heads

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)  # Calculate attention score
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)  # Scale attention score
        weights = tf.nn.softmax(scaled_score, axis=-1)  # Apply softmax to get weights
        output = tf.matmul(weights, value)  # Compute output
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # Transpose to get separate heads

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # Generate query
        key = self.key_dense(inputs)  # Generate key
        value = self.value_dense(inputs)  # Generate value
        query = self.separate_heads(query, batch_size)  # Separate heads in query
        key = self.separate_heads(key, batch_size)  # Separate heads in key
        value = self.separate_heads(value, batch_size)  # Separate heads in value
        attention, weights = self.attention(query, key, value)  # Compute attention
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # Transpose attention
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # Concatenate attention
        output = self.combine_heads(concat_attention)  # Combine heads
        return output

# Transformer block (self-attention + feed-forward)
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)  # Multi-head self-attention
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )  # Feed-forward network
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)  # Layer normalization
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)  # Layer normalization
        self.dropout1 = layers.Dropout(rate)  # Dropout
        self.dropout2 = layers.Dropout(rate)  # Dropout

    def call(self, inputs, training):
        attn_output = self.att(inputs)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)  # Dropout
        out1 = self.layernorm1(inputs + attn_output)  # Add & norm
        ffn_output = self.ffn(out1)  # Feed-forward
        ffn_output = self.dropout2(ffn_output, training=training)  # Dropout
        return self.layernorm2(out1 + ffn_output)  # Add & norm

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(None,))  # Input layer
embedding_layer = layers.Embedding(input_dim=10000, output_dim=embed_dim)  # Embedding layer
x = embedding_layer(inputs)  # Apply embedding
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)  # Transformer block
x = transformer_block(x)  # Apply transformer block
outputs = layers.Dense(10)(x)  # Output layer

model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Create model

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

Image(filename='model.png') 
"""

"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

# XOR inputs and outputs
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

inputs = np.array([[0.5, 0], [0.1, 1], [1.3, 6], [5, 1]])

model = keras.Sequential(
    [
        # the hidden ReLU layers
        layers.Dense(units=8, input_dim=2, activation="relu"),
        layers.Dense(units=8, input_dim=2, activation="relu"),

        # the linear output layer
        layers.Dense(units=1, activation="sigmoid"),
    ]
)


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(inputs, outputs, epochs=2000, verbose=1)


print(model.summary())
plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)


# Evaluate the model
print("\nModel evaluation:")
loss, accuracy = model.evaluate(inputs, outputs, verbose=0)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Make predictions
print("\nModel predictions:")
predictions = model.predict(inputs)
print(predictions.round())
"""
