import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.LSTM import LSTM_Creator
from models.transformer import Transformer_Creator
from models.CNN import CNN_Creator


MODEL_CREATORS = {
    'LSTM': LSTM_Creator,
    'CNN': CNN_Creator,
    'Transformer': Transformer_Creator
}

def main():
    for model_type in MODEL_CREATORS:
        model = MODEL_CREATORS[model_type]().create_model(input_shape=(8, 1), num_classes=2)

        df = pd.read_csv('data.csv')

        X_train = []
        y_train = []
        for i, row in df.iterrows():
            values_flat = np.array([int(char) for char in row['values'] if char in ['0', '1']])

            for i in range(len(values_flat) // 8):
                X_train.append(values_flat[i * 8:(i + 1) * 8].reshape(8, -1))
                y_train.append(row['class'])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        print(X_train.shape)


        if model_type != 'CNN':
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=30)

        accuracy = history.history['accuracy']

        plt.plot(accuracy, label=model_type)

    plt.title('Accuracy of different models')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()