import tflite_runtime.interpreter as tflite
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def predictor(inputdata):

    model_file = os.path.join(BASE_DIR,'coin_classifier.tflite')
    interpreter = tflite.Interpreter(model_path=model_file)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    # here a random data has been provided
    # need to change as our required input
    input_data = np.array(inputdata, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(output_data)
    return output_data
