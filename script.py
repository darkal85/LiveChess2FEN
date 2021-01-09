import sys
import os.path

backend = sys.argv[1]

if backend == "onnx":
    print("onnx backend")

if backend == "trt":
    print("trt backend")

if backend == "keras":
    print("keras backend")
    from keras.applications.xception import preprocess_input as prein_xception
    from keras.models import load_model
    from lc2fen.predict_board import load_image
    from lc2fen.test_predict_board import predict_board

    MODEL_PATH_KERAS = "./selected_models/Xception_last.h5"
    IMG_SIZE_KERAS = 299
    PRE_INPUT_KERAS = prein_xception


def main_keras():
    model = load_model(MODEL_PATH_KERAS)

    def obtain_pieces_probs(pieces):
        predictions = []
        for piece in pieces:
            piece_img = load_image(piece, IMG_SIZE_KERAS, PRE_INPUT_KERAS)
            predictions.append(model.predict(piece_img)[0])
        return predictions

    fen = predict_board(os.path.join("predictions", "test1.jpg"), "BL",
                        obtain_pieces_probs)
    return fen


if __name__ == "__main__":
    if backend == "keras":
        fen = main_keras()
        print(fen)
    else:
        print("other backend; stop")
