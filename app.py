from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from werkzeug.utils import secure_filename
import bcrypt
import jwt
import datetime

app = Flask(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes, allow credentials, and specify the allowed origin
CORS(
    app,
    origins="https://data-leakage-backne.vercel.app/",  # Allow only this origin
    supports_credentials=True,  # Allow credentials (cookies, authorization headers)
)

# Secret key for JWT
app.config['SECRET_KEY'] = 'your-secret-key'  # Replace with a strong secret key

# Configure file upload settings
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MySQL database connection details
MYSQL_HOST="bjpsztsjvidgykjhuovq-mysql.services.clever-cloud.com"
MYSQL_PORT=3306
MYSQL_USER="ujrryv4sv1atae3q"
MYSQL_PASSWORD="nw5aWAxXG5zkFbaBaetL"
MYSQL_DATABASE="bjpsztsjvidgykjhuovq"

# Connect to MySQL database
def get_db_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE
    )
print("hellow world")
# Helper function to check if a file is allowed
ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Helper function to verify passwords
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# Helper function to generate JWT
def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Token expires in 1 hour
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

# Helper function to verify JWT
def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except jwt.ExpiredSignatureError:
        return None  # Token has expired
    except jwt.InvalidTokenError:
        return None  # Invalid token

# Handle preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "https://data-leakage-backne.vercel.app/")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "https://data-leakage-backne.vercel.app/")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
    return response

# API endpoint for user registration
@app.route('/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({}), 200
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    confirm_password = data.get('confirm_password')
    mobile_number = data.get('mobile_number')

    # Validate inputs
    if not username or not password or not confirm_password or not mobile_number:
        return jsonify({"error": "All fields are required"}), 400

    if password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400

    # Hash the password
    hashed_password = hash_password(password)

    # Store the user in the database
    connection = get_db_connection()
    cursor = connection.cursor()
    try:
        query = "INSERT INTO users (username, password, mobile_number) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, hashed_password, mobile_number))
        connection.commit()
        user_id = cursor.lastrowid
        cursor.close()
        connection.close()
        return jsonify({"message": "User registered successfully", "user_id": user_id}), 201
    except mysql.connector.IntegrityError:
        cursor.close()
        connection.close()
        return jsonify({"error": "Username already exists"}), 400

# API endpoint for user login
@app.route('/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({}), 200
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    # Retrieve the user from the database
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM users WHERE username = %s"
    cursor.execute(query, (username,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()

    if not user:
        return jsonify({"error": "Invalid username or password"}), 401

    # Verify the password
    if not verify_password(password, user['password']):
        return jsonify({"error": "Invalid username or password"}), 401

    # Generate JWT
    token = generate_token(user['id'])
    return jsonify({"message": "Login successful", "token": token}), 200

# Protected API endpoint to upload a dataset
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_dataset():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({}), 200
    # Verify the token
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Token is missing"}), 401

    user_id = verify_token(token)
    if not user_id:
        return jsonify({"error": "Invalid or expired token"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the CSV file into a Pandas DataFrame
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({"error": f"Error reading file: {str(e)}"}), 400
        
        # Convert DataFrame to JSON string for storage
        data_json = df.to_json(orient='records')
        
        # Store the dataset in MySQL
        connection = get_db_connection()
        cursor = connection.cursor()
        query = "INSERT INTO datasets (filename, data, user_id) VALUES (%s, %s, %s)"
        cursor.execute(query, (filename, data_json, user_id))
        connection.commit()
        dataset_id = cursor.lastrowid
        cursor.close()
        connection.close()
        
        return jsonify({"message": "File uploaded successfully", "dataset_id": dataset_id}), 200
    else:
        return jsonify({"error": "File type not allowed. Only CSV files are accepted."}), 400

# Protected API endpoint to detect data leakage
@app.route('/detect_leakage/<int:dataset_id>', methods=['GET', 'OPTIONS'])
def detect_leakage(dataset_id):
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({}), 200
    # Verify the token
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Token is missing"}), 401

    user_id = verify_token(token)
    if not user_id:
        return jsonify({"error": "Invalid or expired token"}), 401

    # Retrieve the dataset from MySQL
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT * FROM datasets WHERE id = %s AND user_id = %s"
    cursor.execute(query, (dataset_id, user_id))
    dataset = cursor.fetchone()
    cursor.close()
    connection.close()
    
    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404
    
    # Convert the JSON string back to a Pandas DataFrame
    df = pd.read_json(dataset['data'])
    
    # Check if the dataset has a target column
    if 'target' not in df.columns:
        return jsonify({"error": "Dataset must contain a 'target' column"}), 400
    
    # Data Leakage Detection Logic
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Predict on training and testing data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, train_preds)
    test_accuracy = accuracy_score(y_test, test_preds)
    
    # Check for data leakage
    leakage_detected = train_accuracy > test_accuracy + 0.2  # Threshold for leakage
    
    # Return the results
    return jsonify({
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "leakage_detected": leakage_detected
    }), 200

# Protected API endpoint to list all uploaded datasets
@app.route('/datasets', methods=['GET', 'OPTIONS'])
def list_datasets():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return jsonify({}), 200
    # Verify the token
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Token is missing"}), 401

    user_id = verify_token(token)
    if not user_id:
        return jsonify({"error": "Invalid or expired token"}), 401

    # Retrieve datasets for the authenticated user
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT id, filename FROM datasets WHERE user_id = %s"
    cursor.execute(query, (user_id,))
    datasets = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify(datasets), 200

@app.route('/profile', methods=['GET'])
def profile():
    user_id = request.args.get('user_id')
    if not user_id or not user_id.isdigit():
        return jsonify({"error": "Invalid user ID"}), 400

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    query = "SELECT id, username,mobile_number FROM users WHERE id = %s"  # Updated query
    cursor.execute(query, (user_id,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()  

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify(user), 200


# Run the Flask app
if __name__ == '__main__':
    if os.environ.get('ENV') == 'production':
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    else:
        app.run(debug=True)
