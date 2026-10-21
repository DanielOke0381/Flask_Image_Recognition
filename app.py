"""Flask Application for Image Recognition."""

from flask import Flask, render_template, request
from PIL import UnidentifiedImageError
from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Render home page."""
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def predict_image_file():
    """Process uploaded digit image and render prediction result."""
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or file.filename == "":
                return render_template("result.html", err="File cannot be processed.")

            img = preprocess_img(file.stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))

        except (UnidentifiedImageError, KeyError, ValueError, AttributeError, Exception):
            return render_template("result.html", err="File cannot be processed.")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(port=9000, debug=True)
