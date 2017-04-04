import tensorflow as tf

#Node test
print("NODE TEST:")
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

#Session test
print("SESSION TEST:")
sess = tf.Session()
print(sess.run([node1, node2]))

#Adding node test
print("ADDING NODE TEST:")
node3 = tf.add(node1, node2)
print("Node3: ", node3)
print("Session run of Node3: ", sess.run(node3))

#Test of placeholders   
print("TEST OF PLACEHOLDERS:")
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2,4]}))

#More operations
print("MORE OPERATIONS:")
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3, b:4.5}))

#Variables
print("VARIABLES")
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

#loss function
print("LOSS FUNCTION")
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

#fix loss
print("FIX LOSS")
fixW = tf.assign(W, [-1.])
fixB = tf.assign(b, [1.])
sess.run([fixW, fixB])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

#Machine Learning
print("MACHINE LEARNING")
optimizer = tf.train.GradientDescentOptimizer(0.01)
train= optimizer.minimize(loss)
sess.run(init)
for i in range(100000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([W,b]))
