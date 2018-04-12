import tflearn
import tensorflow as tf

X = [
    [0,0],[0,1],[1,0],[1,1]
]

Y = [
    [0],[1],[1],[0]
]

weights = tflearn.initializations.uniform(minval = -1, maxval = 1)

# Input layer
net = tflearn.input_data(
        shape = [None, 2],
        name = 'my_input'
)

# Hidden layers
net = tflearn.fully_connected(net, 4,
        activation = 'sigmoid',
        weights_init = weights
)
net = tflearn.fully_connected(net, 3,
        activation = 'sigmoid',
        weights_init = weights
)

# Output layer
net = tflearn.fully_connected(net, 1,
        activation = 'sigmoid', 
        weights_init = weights,
        name = 'my_output'
)

# Configuration
net = tflearn.regression(net,
        learning_rate = 2,
        optimizer = 'sgd',
        loss = 'mean_square'
)

model = tflearn.DNN(net)
model.fit(X, Y, 5000)

print("1 XOR 0 = %f" % model.predict([[1,0]]).item(0))
print("1 XOR 1 = %f" % model.predict([[1,1]]).item(0))
print("0 XOR 1 = %f" % model.predict([[0,1]]).item(0))
print("0 XOR 0 = %f" % model.predict([[0,0]]).item(0))

with net.graph.as_default():
     del tf.get_collection_ref(tf.GraphKeys.TRAIN_OPS)[:]

model.save('xor.tflearn')
