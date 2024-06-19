
#functional APi Demo
from keras.models import Model

model = Model(inputs = x ,outputs = [output1,output2])

from keras.layers import *

x = Input(shape=(3,))

hidden1 = Dense(128,activation='relu')(x)
hidden2 = Dense(64,activation='relu')(hidden1)

output1 = Dense(1,activation='linear')(hidden2)
output2 = Dense(1,activation='sigmoid')(hidden2)

model.summary()

from keras.utils import plot_model
plot_model(model,show_shapes=True)


#with Multiple Input
from keras.layers import *
from keras.models import Model

# define two sets of inputs
inputA = Input(shape=(32,))
inputB = Input(shape=(128,))

# the first branch operates on the first input
x = Dense(8, activation="relu")(inputA)
x1 = Dense(4, activation="relu")(x)

# the second branch opreates on the second input
y = Dense(64, activation="relu")(inputB)
y1 = Dense(32, activation="relu")(y)
y2 = Dense(4, activation="relu")(y1)

# combine the output of the two branches
combined = concatenate([x1, y2])

# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(2, activation="relu")(combined)
z1 = Dense(1, activation="linear")(z)

# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[inputA, inputB], outputs=z1)

from keras.utils import plot_model
plot_model(model)

