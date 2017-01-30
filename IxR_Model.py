import numpy as np


# class IBR_Model(IteratedModel):
#     functions specific for ibr
#         selection
#

class IteratedModel(object):
    def __init__(self, B, sender_cost=None, receiver_cost=None, receiver_prior=None):
        self.B = B
        self.level = 0
        self.sender = []
        self.receiver = []
        if sender_cost is not None:
            self.sender_cost = sender_cost
        else:
            self.sender_cost = np.zeros(B.shape[1])
        if receiver_cost is not None:
            self.receiver_cost = receiver_cost
        else:
            self.receiver_cost = np.zeros(B.shape[1])
        if receiver_prior is not None:
            self.receiver_prior = receiver_prior
        else:
            self.receiver_prior = np.ones(B.shape[0])

    def next_level_reasoning(self):
        if self.level == 0:
            self.sender.insert(self.level, [normalize(self.B)])  # add sender cost
            self.receiver.insert(self.level, [normalize(self.B.T)])
        else:
            self.sender.insert(self.level, [])
            self.receiver.insert(self.level, [])
            for strategy in self.receiver[self.level - 1]:
                strategy_to_respond_to = np.copy(strategy.T) - self.sender_cost
                self.sender[self.level].append(self.find_best_response(strategy_to_respond_to, "sender"))
            for strategy in self.sender[self.level - 1]:
                # Notice that first we apply the prior and then we apply the cost!
                strategy_to_respond_to = ((np.copy(strategy.T) * self.receiver_prior).T - self.receiver_cost).T
                self.receiver[self.level].append(self.find_best_response(strategy_to_respond_to, "receiver"))
        self.level += 1

    def find_best_response(self, strategy, type):
        # we find the highest values by rows
        max_indexes = np.argmax(strategy, axis=1)
        max_values = np.amax(strategy, axis=1)
        min_values = np.amin(strategy, axis=1)
        for index, value in enumerate(max_values):
            # the highest value is the same as the lowest => all the same value
            if max_values[index] == min_values[index]:
                # a sender with all rows = 0, does not send that message, we leave it as is
                if type == "sender" and max_values[index] == 0.0:
                    continue
                # otherwise we set it uniform. It is always uniform for receiver
                else:
                    strategy[index] = self.get_uniform_row(strategy.shape[1])
            # there are multiple highest values in a row => normalize them and set others to 0
            elif np.count_nonzero(strategy[index] == max_values[index]) != 1:
                # find where they are
                indexes = np.nonzero(strategy[index] == max_values[index])
                # set all to 0
                new_row = np.zeros(strategy.shape[1])
                for high_index in indexes:
                    new_row[high_index] = 1.0
                strategy[index] = row_wise_division(new_row)
            # there is a single highest value
            else:
                new_row = np.zeros(strategy.shape[1])
                new_row[max_indexes[index]] = 1.0
                strategy[index] = new_row
        return strategy

    def get_sender(self):
        return self.sender

    def get_receiver(self):
        return self.receiver

    def get_uniform_row(self, shape):
        return np.ones(shape) * 1 / shape

    def __str__(self):
        # TODO adjust nicely with formatting
        print("Initial setup")
        print(self.B)
        print("-----Sender------:------Receiver-------\n")
        for level in range(0, self.level):
            print(self.sender[level])
            print(self.receiver[level])
            print("\n")
        # TODO fix to string hack
        return ""


def normalize(B):
    return np.apply_along_axis(row_wise_division, 1, B)


def row_wise_division(row):
    sum = np.sum(row)
    if sum == 0.0:
        return row
    return row * (1 / sum)


import unittest


class TestModels(unittest.TestCase):
    def test_asymmetric_evil_sender(self):
        B = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])
        receiver_cost = np.array([0, 0, 0, 1])
        sender_cost = np.array([0, 0, 0, -1])
        # works, the sender does not send the costly message of the receiver
        iterated_model = IteratedModel(B, sender_cost, receiver_cost, None)
        for x in range(0, 5):
            iterated_model.next_level_reasoning()
        print(iterated_model)
        np.testing.assert_array_equal(iterated_model.get_receiver()[4][0], np.array([[1., 0.],
                                                                                     [1., 0.],
                                                                                     [0.5, 0.5],
                                                                                     [0., 1.]]))
        np.testing.assert_array_equal(iterated_model.get_sender()[4][0], np.array([[1.0 / 3, 1.0 / 3, 0., 1.0 / 3],
                                                                                   [0., 0., 0., 1.]]))

    def test_symmetric_nice_sender(self):
        B = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])
        receiver_cost = np.array([0, 0, 0, 1])
        sender_cost = receiver_cost + np.array([0, 1, 0, 0])
        # works, the sender does not send the costly message of the receiver
        iterated_model = IteratedModel(B, sender_cost, receiver_cost, None)
        for x in range(0, 5):
            iterated_model.next_level_reasoning()
        np.testing.assert_array_equal(iterated_model.get_receiver()[4][0], np.array([[ 1. ,  0. ],
                                                                                     [ 0.5,  0.5],
                                                                                     [ 0. ,  1. ],
                                                                                     [ 0.5,  0.5]]))
        np.testing.assert_array_equal(iterated_model.get_sender()[4][0], np.array([[ 1.,  0.,  0.,  0.],
                                                                                   [ 0.,  0.,  1.,  0.]]))

    def test_symmetric(self):
        B = np.array([[1, 1, 0, 0],
                      [0, 0, 1, 1]])
        sender_cost = np.array([0, 1, 0, 0])
        receiver_cost = np.array([0, 0, 0, 1])
        iterated_model = IteratedModel(B, sender_cost, receiver_cost, None)
        for x in range(0, 5):
            iterated_model.next_level_reasoning()

    def test_asymmetric(self):
        B = np.array([[1, 1, 0],
                      [0, 0, 1]])
        sender_cost = np.array([0, 0, 0])
        receiver_cost = np.array([0, 0, 1])
        iterated_model = IteratedModel(B, sender_cost, receiver_cost, None)
        for x in range(0, 5):
            iterated_model.next_level_reasoning()

    def test_hypo(self):
        B = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
        iterated_model = IteratedModel(B, [0, 0.1, 0.2], None, [0.1, 0.2, 0.3])
        for x in range(0, 5):
            iterated_model.next_level_reasoning()

    def test_normalize(self):
        B = np.array([[1, 0],
                      [1, 1]])
        norm_B = np.array([[1, 0.],
                           [0.5, 0.5]])
        np.testing.assert_array_equal(normalize(B), norm_B)
        B = np.array([[1, 0],
                      [0, 0]])
        norm_B = np.array([[1, 0.],
                           [0, 0]])
        np.testing.assert_array_equal(normalize(B), norm_B)

    def test_single_solution_ibr(self):
        B = np.array([[1, 0],
                      [1, 1]])
        iterated_model = IteratedModel(B)
        for x in range(0, 5):
            iterated_model.next_level_reasoning()
        np.testing.assert_array_equal(iterated_model.get_sender()[0][0], np.array([[1., 0.],
                                                                                   [0.5, 0.5]]))
        np.testing.assert_array_equal(iterated_model.get_sender()[1][0], np.array([[1., 0.],
                                                                                   [0., 1.]]))
        np.testing.assert_array_equal(iterated_model.get_receiver()[0][0], np.array([[0.5, 0.5],
                                                                                     [0., 1.]]))
        np.testing.assert_array_equal(iterated_model.get_receiver()[1][0], np.array([[1., 0.],
                                                                                     [0, 1.]]))

    def test_multiple_solution_ibr(self):
        B = np.array([[1, 1],
                      [1, 1]])
        iterated_model = IteratedModel(B, np.array([0, 0.1]), np.array([0, 0]), np.array([0.51, 0.49]))
        for x in range(0, 5):
            iterated_model.next_level_reasoning()
        np.testing.assert_array_equal(iterated_model.get_sender()[0][0], np.array([[0.5, 0.5],
                                                                                   [0.5, 0.5]]))
        np.testing.assert_array_equal(iterated_model.get_sender()[1][0], np.array([[1., 0.],
                                                                                   [1., 0.]]))
        np.testing.assert_array_equal(iterated_model.get_receiver()[0][0], np.array([[0.5, 0.5],
                                                                                     [0.5, 0.5]]))
        np.testing.assert_array_equal(iterated_model.get_receiver()[1][0], np.array([[1., 0.],
                                                                                     [1., 0.]]))


if __name__ == '__main__':
    unittest.main()
