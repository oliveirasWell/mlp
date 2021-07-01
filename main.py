import itertools

import numpy as np


def add_bias(matrix_input):
    ones = np.ones((matrix_input.shape[0], 1))
    concatenated = np.concatenate((matrix_input, ones), axis=1)
    return concatenated


def activation_function(matrix_input):
    return np.tanh(matrix_input)


def der_activation_function(y):
    return 1 - np.tanh(y) * np.tanh(y)


def main():
    w_hidden_layer = np.array([
        [0.2, 0.2, -0.1],
        [-0.1, 0.3, 0.3],
        [0.1, 0.9, 0.1],
    ])

    w_output_layer = np.array([
        [0.1, -0.1, -0.1, 0.2],
        [0.5, 0.2, 1.1, -0.1]
    ])

    LEARNING_RATE = 0.2

    inputs = [
        np.array([[0.1, 0.7, 1]]),
        # np.array([[0.2, 0.3, 1]]),
        # np.array([[0.3, 0.4, 1]]),
        # np.array([[0.4, 0.5, 1]]),
        # np.array([[0.5, 0.6, 1]]),
    ]

    classes = [
        [0.2, 1],
    ]

    for input_element, classes_element in itertools.zip_longest(inputs, classes):
        print("input_element")
        print(input_element)

        print("classes_element")
        print(classes_element)

        activation_function_input = np.matmul(input_element, w_hidden_layer.T)

        print("activation_function_input")
        print(activation_function_input)

        hidden_layer_result = activation_function(activation_function_input)

        print("hidden_layer_result")
        print(hidden_layer_result)

        hidden_layer_result_with_bias = add_bias(hidden_layer_result)

        print("hidden_layer_result_with_bias")
        print(hidden_layer_result_with_bias)

        output_layer_input = np.matmul(hidden_layer_result_with_bias, w_output_layer.T)

        print("output_layer_input")
        print(output_layer_input)

        output_layer_result = activation_function(output_layer_input)

        print("output_layer_result")
        print(output_layer_result)

        errors_output_layer = np.subtract(np.array([classes_element]), output_layer_result)

        print("errors_output_layer")
        print(errors_output_layer)

        derived_external_result = der_activation_function(output_layer_result)
        print("derived_external_result")
        print(derived_external_result)

        diag_errors = np.diag(errors_output_layer[0])
        print("diag_errors")
        print(diag_errors)

        output_layer_gradient = np.matmul(derived_external_result, diag_errors)

        print("output_layer_gradient")
        print(output_layer_gradient)

        hidden_layer_local_gradient = der_activation_function(hidden_layer_result_with_bias)

        print("hidden_layer_local_gradient")
        print(hidden_layer_local_gradient)

        w_output_layer_without_bias = np.delete(w_output_layer, 1, 1)

        print("w_output_layer_without_bias")
        print(w_output_layer_without_bias)

        gradient_hidden_layers_coonections = np.matmul(output_layer_gradient, w_output_layer)

        print("gradient_hidden_layers_coonections")
        print(gradient_hidden_layers_coonections)

        hidden_layer_gradients = hidden_layer_local_gradient * gradient_hidden_layers_coonections

        print("hidden_layer_gradients")
        print(hidden_layer_gradients)

        # update weights
        new_output_weights = w_output_layer + (LEARNING_RATE * hidden_layer_result_with_bias.T * output_layer_gradient).T
        w_output_layer = new_output_weights

        print("new_output_weight")
        print(new_output_weights)

        new_hidden_weight = w_hidden_layer + (LEARNING_RATE * input_element.T * np.delete(hidden_layer_gradients, 1, 1)).T

        print("new_hidden_weight")
        print(new_hidden_weight)

        w_hidden_layer = new_hidden_weight

        print("================================================")


if __name__ == '__main__':
    main()
