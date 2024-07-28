import os
from flask import (
    Flask,
    flash,
    request,
    redirect,
    render_template,
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from classifier import classifier


load_dotenv()

STATIC_FOLDER = "static"
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":

        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            flash("No selected file, please select a file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(upload_image_path)

            predicted_label = classifier(STATIC_FOLDER + "/models/" + "model.tflite", upload_image_path)

            return render_template(
                "result.html", image=f"uploads/{filename}", prediction=predicted_label
            )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
