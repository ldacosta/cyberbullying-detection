import time
import sys
import numpy as np


class CyberbullyingFullyConnectedNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1):
        """Create a CyberbullyingFullyConnectedNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of THREAT/CLEAN labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        # reviews
        review_vocab = set()
        for review in reviews:
            review_vocab = review_vocab.union(set(review.lower().split(' ')))
        self.review_vocab = list(review_vocab)
        # labels
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        self.label_vocab = list(label_vocab)
        #
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        # TODO: populate self.word2index with indices for all the words in self.review_vocab
        #       like you saw earlier in the notebook
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, word in enumerate(self.label_vocab):
            self.label2index[word] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        # initialize self.weights_1_2 as a matrix of random values.
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.rand(self.hidden_nodes, self.output_nodes)

        # Create the hidden layer, a two-dimensional matrix with shape
        #       1 x hidden_nodes, with all values initialized to zero
        self.layer_1 = np.zeros((1, self.hidden_nodes))

    def get_target_for_label(self, label):
        if label == 'CLEAN':
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):
        def safe_word_2_index(word) -> int:
            try:
                return self.word2index[word]
            except KeyError:
                return -1

        training_reviews = []
        for review_raw_str in training_reviews_raw:
            review_raw = review_raw_str.split(' ')
            # l = [i for i in [safe_word_2_index(word) for word in review_raw] if i != -1]
            reviews_on = list(set([i for i in [safe_word_2_index(word) for word in review_raw] if i != -1]))
            # reviews_on = list(set([self.word2index[word] for word in review_raw]))
            training_reviews.append(reviews_on)

        # make sure out we have a matching number of reviews and labels
        assert (len(training_reviews) == len(training_labels))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # TODO: Get the next review and its correct label
            reviews_on = training_reviews[i]

            # print("len(reviews_on) = {}".format(len(reviews_on)))
            label = training_labels[i]

            # TODO: Implement the forward pass through the network.
            #       That means use the given review to update the input layer,
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            #
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.

            # calculate hidden layer data
            # hidden_in = np.dot(self.layer_0, self.weights_0_1)
            hidden_in = self.layer_1 * 0  # I want it to have the same shape
            # print("hidden_in.shape = {}".format(hidden_in.shape))
            for index in reviews_on:
                # print("self.weights_0_1.shape = {}".format(self.weights_0_1.shape))
                # print("self.weights_0_1[{}].shape = {}".format(index, self.weights_0_1[index].shape))
                hidden_in += (self.weights_0_1[index])

            self.layer_1 = hidden_in  # because activation == Id.
            # calculate output layer data
            output_in = np.dot(self.layer_1, self.weights_1_2)
            output_out = self.sigmoid(output_in)

            # TODO: Implement the back propagation pass here.
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you
            #       learned in class.

            # Let's calculate errors:
            # at the output layer
            error = self.get_target_for_label(label) - output_out
            # print("error = {}".format(error))
            error_term = error * self.sigmoid_output_2_derivative(self.sigmoid(output_in))
            # print("error_term = {}".format(error_term))
            # at the hidden layer
            # hidden_error = np.dot(error_term, self.weights_1_2.T) # * hidden_in
            hidden_error_term = np.dot(error_term,
                                       self.weights_1_2.T) * 1  # because activate function == Id, so derivative == 1
            # print("hidden_error_term = {}".format(hidden_error_term))
            # and now let's update weights:
            self.weights_1_2 += self.learning_rate * np.dot(self.layer_1.T, error_term)
            # print("BEFORE ==> non-zeros = {}".format(np.count_nonzero(self.weights_0_1)))
            # print("self.layer_0 has {} non-zeros; self.learning_rate = {}".format(np.count_nonzero(self.layer_0), self.learning_rate))
            #             layer_0 = np.zeros((1, self.input_nodes))
            #             layer_0[0][reviews_on] = 1
            #             self.weights_0_1 += self.learning_rate * np.dot(layer_0.T,hidden_error_term)

            for index in reviews_on:
                self.weights_0_1[index] += hidden_error_term[
                                               0] * self.learning_rate  # update input-to-hidden weights with gradient descent step

            # print("AFTER ==> non-zeros = {}".format(np.count_nonzero(self.weights_0_1)))
            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error
            #       is less than 0.5. If so, add one to the correct_so_far count.
            only_neuron_error = np.abs(error[0, 0])
            # print("error's shape is {}, only_neuron_error == {}".format(error.shape, only_neuron_error))
            if only_neuron_error < 0.5:
                # print("correct_so_far++")
                correct_so_far += 1
            # if only_neuron_error > 0.85:
            #    print("dafuq error is {}".format(only_neuron_error))

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(training_reviews_raw)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i + 1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%")
            if (i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if (pred == testing_labels[i]):
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i / float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i + 1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i + 1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction
        #             might come from anywhere, so you should convert it
        #             to lower case prior to using it.



        review_raw = review.lower().split(' ')
        # print("len(review_raw) = {}".format(len(review_raw)))
        reviews_on = []
        for word in review_raw:
            try:
                reviews_on.append(self.word2index[word])
            except KeyError:
                pass

        # reviews_on = [self.word2index[word] for word in review_raw] # contains the indices for words found in the review.
        # calculate hidden layer data
        # hidden_in = np.dot(self.layer_0, self.weights_0_1)
        hidden_in = self.layer_1 * 0  # I want it to have the same shape
        # print("hidden_in.shape = {}".format(hidden_in.shape))
        for index in reviews_on:
            # print("self.weights_0_1.shape = {}".format(self.weights_0_1.shape))
            # print("self.weights_0_1[{}].shape = {}".format(index, self.weights_0_1[index].shape))
            hidden_in += (self.weights_0_1[index])

        self.layer_1 = hidden_in  # because activation == Id.
        # calculate output layer data
        output_in = np.dot(self.layer_1, self.weights_1_2)
        output_out = self.sigmoid(output_in)

        #         # create input
        #         self.update_input_layer(review.lower())
        #         # calculate hidden layer data
        #         hidden_in = np.dot(self.layer_0, self.weights_0_1)
        #         hidden_out = hidden_in # because activation == Id.
        #         # calculate output layer data
        #         output_in = np.dot(hidden_out, self.weights_1_2)
        #         output_out = self.sigmoid(output_in)

        # TODO: The output layer should now contain a prediction.
        #       Return `POSITIVE` for predictions greater-than-or-equal-to `0.5`,
        #       and `NEGATIVE` otherwise.
        if output_out >= 0.5:
            return 'CLEAN'
        else:
            return 'THREAT'
