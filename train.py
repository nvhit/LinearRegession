import os
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import csv
import json
import tempfile
import requests
import numpy as np
import tensorflow as tf


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fileInput", default='', type=str)
    args = parser.parse_args()
    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

    with open(args.fileInput, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                xs = np.array(np.array(row, dtype=float))
                line_count += 1
            else:
                ys = np.array(np.array(row, dtype=float))
                line_count += 1
        print(f'Processed {line_count} lines.')

    print(xs)
    print(ys)

    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd',
                  loss='mean_squared_error', metrics='mse')
    history = model.fit(xs, ys, epochs=1000, verbose=0)
    print("Finished training the model")

    print(model.predict([12.0]))

    file_pb = "./output/out_linear_pb"
    file_h5 = "./output/out_linear.h5"

    model.save(file_h5)

    tf.saved_model.save(model, file_pb)

    xs = np.array([[9.0], [50.0], [2.0], [3.0]])
    data = json.dumps({"signature_name": "serving_default", "instances": xs.tolist()})
    print(data)

