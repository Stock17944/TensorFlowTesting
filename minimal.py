import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()
losstotal = []
attempt = 0
attemptl = []

#Variables
W = tf.Variable([44.], tf.float32)
b = tf.Variable([213.], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

#Loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

#Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(10000):
    sess.run(train, {x:[1,2,3,4,5], y:[0,-1,-2,-3,-4]})
    attempt += 1
    attemptl.append(attempt)
    losstotal.append(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
    print("Attempts left: {}".format(10000-attempt))
    
print("First 10 attempts loss: {}\nLast attempt loss: {}\nWinner W variable: {}\nWinner b variable: {}".format(losstotal[0:10], losstotal[-1], sess.run(W), sess.run(b)))

plt.plot(attemptl, losstotal)
plt.ylabel("Loss")
plt.xlabel("Attempt #")
plt.show()
