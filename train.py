from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
import csv
import json
import tempfile
import requests
import numpy as np
import tensorflow as tf

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



    xs = np.array([[9.0], [50.0], [2.0], [3.0]])
    data = json.dumps({"signature_name": "serving_default", "instances": xs.tolist()})
    print(data)

