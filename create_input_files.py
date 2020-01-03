from utils import create_input_files, train_word2vec_model

if __name__ == '__main__':
    create_input_files(csv_folder='/users5/yjtian/tyj/demo/Hierarchical-Attention-Network/yahoo_answers_csv',
                       output_folder='/users5/yjtian/tyj/demo/HAN/data',
                       sentence_limit=15,
                       word_limit=20,
                       min_word_count=5)

    train_word2vec_model(data_folder='/users5/yjtian/tyj/demo/HAN/data',
                         algorithm='skipgram')
