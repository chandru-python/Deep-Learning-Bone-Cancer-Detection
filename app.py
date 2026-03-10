from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Flask App Config
app = Flask(__name__, static_folder='static')
app.secret_key = "dyuiknbvcxswe678ijc6i"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained CNN model
model = load_model('trained_model_CNN.h5')

# Class labels (adjust based on your training)
burn_classes = ['malignant', 'Normal']
img_width, img_height = 224, 224

# DB Connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="bone"
    )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        try:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                flash("Login successful!", "success")
                return redirect(url_for("prediction"))
            else:
                flash("Invalid email or password.", "danger")
        finally:
            cursor.close()
            conn.close()
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (name, email, phone, password) VALUES (%s, %s, %s, %s)",
                (name, email, phone, hashed_password)
            )
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except mysql.connector.Error as err:
            flash(f"Registration failed: {err}", "danger")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if "user_id" not in session:
        flash("Please login first to use prediction feature.", "warning")
        return redirect(url_for("login"))

    prediction_result = None
    image_path = None

    if request.method == "POST":
        if 'image' not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Process and predict
            img = image.load_img(image_path, target_size=(img_width, img_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x.astype('float32') / 255.0
            prediction = model.predict(x)
            predicted_class = np.argmax(prediction)
            prediction_result = burn_classes[predicted_class]

    if image_path:
        image_filename = os.path.basename(image_path)
    else:
        image_filename = None

    return render_template("predict.html", prediction=prediction_result, image=image_filename)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
