#%reset -f

# Import `tensorflow`
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#generate data
n_indiv = 1000
x = np.random.uniform(-5,5,[n_indiv,2])
y = [ np.linalg.norm(np.array([a,b])-np.array([0,0])) > 2 for [a,b] in x] 
y = np.logical_xor(y,[ np.linalg.norm(np.array([a,b])-np.array([3,3])) > 1 for [a,b] in x] )
y = np.reshape(y,[n_indiv,1])

print('This is the task to learn')
plt.scatter(x[:,0],x[:,1],c=['blue' if i == True else 'orange' for i in y], marker="+")
    
# Initialize two constants
X = tf.placeholder("float")
Y = tf.placeholder(tf.float32)



# Parameters
learning_rate = 0.00001
training_epochs = 500
batch_size = 10
display_step = 1

# Network Parameters
n_hidden_1 = 100 # 1st layer number of neurons
n_hidden_2 = 2 # 2nd layer number of neurons
n_input = 8 # MNIST data input (img shape: 28*28)
n_classes = 1 # MNIST total classes (0-9 digits)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x):
    layer_1 = tf.add(tf.nn.leaky_relu(tf.matmul(x, weights['h1'])), biases['b1'])
    layer_2 = tf.add(tf.nn.leaky_relu(tf.matmul(layer_1, weights['h2'])), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer,layer_2

# Construct model
    
p2 = tf.pow(X,2)
s2 = tf.sinh(X)
c2 = tf.cos(X)
logits, layer_2 = multilayer_perceptron(tf.concat([X,p2,s2,c2],1))

# Define loss and optimizer
sigmoid_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
loss_op = tf.reduce_mean(sigmoid_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

sigmoid = tf.sigmoid(logits)
pred = tf.round(sigmoid)
correct_prediction = tf.equal(pred, Y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print_op = tf.Print(pred,[pred])

# Initializing the variables
init = tf.global_variables_initializer()

sess =  tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_indiv/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        ids = np.random.randint(0,n_indiv,batch_size)
        batch_x =  [x[i] for i in ids]
        batch_y =  [y[i] for i in ids]
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c, p = sess.run([train_op, loss_op, print_op], feed_dict={X: batch_x,Y: batch_y})
        d = np.round(np.concatenate((batch_x,np.reshape(np.sum(batch_x,1),np.shape(batch_y)),batch_y,p),axis=1),1)
        #print(d)
        #print(p)
        
        # Compute average loss
        avg_cost += c / total_batch
        
    vpred,cpred,sig, l = sess.run([pred,correct_prediction,sigmoid, layer_2], feed_dict={X: x,Y: y})

    l = l - np.mean(l,0)
    l = l / np.std(l,0)

    #plt.scatter(x[0:n_indiv,0],x[0:n_indiv,1],c=sig*100)#c=['green' if i == True else 'red' for i in cpred]) #['green' if  np.int(i[0]) == 1 else 'red'for i in vpred])#
    plt.scatter(l[0:n_indiv,0],l[0:n_indiv,1],c=['green' if i == True else 'red' for i in cpred], marker="+") #['green' if  np.int(i[0]) == 1 else 'red'for i in vpred])#
    #plt.savefig('train6/foo'+str(epoch)+'.png')
    plt.show()
    plt.clf()


    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    # Test model
    print("Accuracy:", accuracy.eval(session=sess,feed_dict={X: x, Y: y}))
print("Optimization Finished!")


# Calculate accuracy
# todo
#

#p = sess.run([print_op], feed_dict={X: [[0,-1]],Y:[[1]] })
#print(p)
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#scatter = ax.scatter(x[0:n_indiv,0],x[0:n_indiv,1],c=['red' if i == 1 else 'blue' for i in y])
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#plt.colorbar(scatter)
#
#fig.show()






