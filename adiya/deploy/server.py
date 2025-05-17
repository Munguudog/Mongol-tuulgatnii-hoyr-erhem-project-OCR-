from flask import Flask, request, jsonify
from io_utils import load_image
from preprocessing import preprocess_image
import segmentation as seg
from models import predict_text, invert_and_pad
from postprocessing import reconstruct_text
import numpy as np

app = Flask(__name__)


def to_serializable(val):
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    return val

@app.route('/ocr', methods=['POST'])
def ocr():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        image = load_image(image_file)

        # Step 1: Preprocess
        processed_image = preprocess_image(image)

        # # Step 2: Segment
        line_bboxes = seg.line_segmenter(processed_image)

        results = []
        texts = ""
        for line_bbox in line_bboxes:
            # Step 3: Word segmentation within line
            word_bboxes = seg.segment_words_from_line(
                gray=processed_image,  # assuming segment_words_from_line needs grayscale or binary
                binary=processed_image,
                line_bbox=line_bbox
            )

            line_result = []
            for word_bbox in word_bboxes:

                x1, y1, x2, y2 = word_bbox
                cropped = processed_image[y1:y2, x1:x2]
                # cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cropped = invert_and_pad(cropped)


                prediction = predict_text(cropped)
                texts+=prediction+" "
                line_result.append({
                    "text": prediction,
                    "bbox": [to_serializable(x) for x in word_bbox]
                })
            texts+="\n"

            results.append({
                "line_bbox": [to_serializable(x) for x in line_bbox],
                "words": line_result
            })
            print(results)


        return jsonify({"text": texts, "results": results})
    except Exception as e:
        return f"error:  {e}"

@app.route('/raw', methods=['POST'])
def raw():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        image = load_image(image_file)

        prediction = predict_text(image, rotate = False)



        return jsonify({"text": prediction})
    except Exception as e:
        return f"error:  {e}"


if __name__ == '__main__':
    app.run(debug=True)
