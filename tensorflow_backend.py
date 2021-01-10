from keras.applications.xception import preprocess_input as prein_xception
from keras.models import load_model
from lc2fen.predict_board import load_image
from lc2fen.test_predict_board import predict_board

_MODEL_PATH_KERAS_ = "./selected_models/Xception_last.h5"
_IMG_SIZE_KERAS_ = 299
_PRE_INPUT_KERAS_ = prein_xception


def predict_fen(img_path):
    model = load_model(_MODEL_PATH_KERAS_)

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, _IMG_SIZE_KERAS_, _PRE_INPUT_KERAS_)
            predictions.append(model.predict(piece_img)[0])
        return predictions

    fen = predict_board(img_path, "BL", obtain_pieces_probs)
    return fen
