from sklearn.model_selection import train_test_split


def split(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0
    )
    print("Shape of x_train:", x_train.shape)
    print("Shape of x_test:", x_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    return x_train, x_test, y_train, y_test

def modelVariables(x_train, x_test, y_train, y_test, tokens):
    token_eng, token_hindi = tokens
    max_length_input = x_train.shape[1]
    max_length_output = y_train.shape[1]
    input_vocab_size = len(token_eng.word_index) + 1
    output_vocab_size = len(token_hindi.word_index) + 1

    print("max_length_input:", max_length_input)
    print("max_length_output:", max_length_output)
    print("input_vocab_size:", input_vocab_size)
    print("output_vocab_size:", output_vocab_size)

    return max_length_input, max_length_output, input_vocab_size, output_vocab_size

