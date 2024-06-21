# Import the necessary packages
import os
import io
import base64
import uuid
from datetime import date
import pickle
import joblib
import shutil
import pandas as pd
import numpy as np
from flask import Flask, render_template, redirect, url_for, request, session, Response, send_file, flash
from imblearn.over_sampling import RandomOverSampler
from werkzeug.utils import secure_filename
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Create a Flask web application instance
app = Flask(__name__)

# Set a secret key for the session
app.secret_key = '1234'

# Configure Flask app settings
app.config["CACHE_TYPE"] = "null"  # Disable caching
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable file caching


import mysql.connector  # Import the MySQL connector library
from werkzeug.security import generate_password_hash, check_password_hash
# Configure MySQL connection
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Junlamlee@2201'
app.config['MYSQL_DB'] = 'tp055697fyp'

# Initialize MySQL
mysql = mysql.connector.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)

@app.route('/home', methods=['GET', 'POST'])
def home():
	return redirect(url_for('input'))


#-----------------------------------------------------------------------------------------------------------
@app.route('/dashboard_createuser', methods=['GET', 'POST'])
def createuser():
    error = None  # Initialize the error message
     # Check if user is logged in
    if 'user_id' not in session:
        return redirect(url_for('input'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists in the database
        cursor = mysql.cursor(dictionary=True)
        cursor.execute("SELECT * FROM dashboard WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            error = 'Username already exists. Please choose a different one.'
            return render_template('dashboard_createuser.html', error=error)
        else:
            # Hash the password before storing it in the database
            password_hash = generate_password_hash(password)
            try:
                # Insert the user into the database
                cursor = mysql.cursor()
                cursor.execute("INSERT INTO dashboard (username, password_hash) VALUES (%s, %s)", (username, password_hash))
                mysql.commit()
                cursor.close()
                msg = ('User created successfully') 
                return render_template('dashboard_createuser.html', msg=msg)
            except Exception as e:
                print("Error creating user:", str(e))
                return render_template('dashboard_createuser.html', error=error)
    return render_template('dashboard_createuser.html')



@app.route('/input', methods=['GET', 'POST'])
def input():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = mysql.cursor(dictionary=True)
        cursor.execute("SELECT * FROM dashboard WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        if user and check_password_hash(user['password_hash'], password):
            # Password is correct, store user data in session
            session['user_id'] = user['id']
            flash('Login successful', 'success')
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password'
    return render_template('input.html', error=error)



@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('input'))  # Redirect to the login page after logging out



#-----------------------------------------------------------------------------------------------------------
# Route for the upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if request.form['sub'] == 'Upload':
            savepath = r'upload/'
            if 'dataset' in request.files:
                uploaded_files = request.files.getlist('dataset')
                file_names = []
                for upload_csv in uploaded_files:
                    if upload_csv and upload_csv.filename.endswith('.csv'):  # Check if the file ends with .csv
                        today_date = date.today().strftime("%d-%m-%Y")  # Get today's date in the desired format
                        unique_filename = secure_filename(f"{today_date}_{upload_csv.filename.replace('.csv', '.csv')}")
                        file_path = os.path.join(savepath, unique_filename)
                        upload_csv.save(file_path)
                        file_names.append(unique_filename)
                    else:
                        return render_template('upload.html', mgs="No valid CSV files uploaded.")
                if file_names:
                    return render_template('upload.html', mgs="Datasets Uploaded: " + ', '.join(file_names))
                else:
                    return render_template('upload.html', mgs="Datasets Upload Error, Please Try Again.")
    return render_template('upload.html')



#-----------------------------------------------------------------------------------------------------------
# Route for the dataset page
@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    # Get a list of available CSV files in the 'upload' directory
    dataset_files = [file for file in os.listdir('upload') if file.endswith('.csv')]
    # Check if a file is selected by the user
    selected_file = request.form.get('selected_file')
    selected_data = None
    shape_info = None  # Initialize shape information
    data_types = None  # Initialize data types information
    
    # If a file is selected, read and display its content
    if selected_file:
        selected_data = pd.read_csv(os.path.join('upload', selected_file))
        shape_info = selected_data.shape  # Get the shape information
        data_types = selected_data.dtypes  # Get the data types of columns
    
    return render_template('dataset.html', dataset_files=dataset_files, selected_data=selected_data, shape_info=shape_info, data_types=data_types)




#-----------------------------------------------------------------------------------------------------------
# Route for the clean page
concatenated_df = pd.DataFrame()  # Store the concatenated DataFrame globally

@app.route('/clean', methods=['GET', 'POST'])
def clean():
    # List available CSV files for selection
    dataset_files = [f for f in os.listdir('upload') if f.endswith('.csv')]
    
    if request.method == 'POST':
        selected_files = request.form.getlist('selected_file')
        if selected_files:
            # Read the first selected CSV file to get its columns
            first_file_path = os.path.join('upload', selected_files[0])
            if os.path.exists(first_file_path):
                first_df = pd.read_csv(first_file_path) 
                common_columns = first_df.columns.tolist()
                
                # Read and concatenate selected CSV files while ensuring they have the same columns
                dfs = [first_df]  # Initialize with the first DataFrame
                for file_name in selected_files[1:]:
                    file_path = os.path.join('upload', file_name)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                
                        # Check if the current DataFrame has the same columns as the first one
                        if df.columns.tolist() == common_columns:
                            dfs.append(df)
                        else:
                            return render_template('clean.html', dataset_files=dataset_files, error='Selected datasets have different columns.')

                # Concatenate the DataFrames
                global concatenated_df
                concatenated_df = pd.concat(dfs, ignore_index=True, sort=False)
                return redirect(url_for('clean_step1'))  # Redirect to the first step of cleaning
    return render_template('clean.html', dataset_files=dataset_files)




@app.route('/clean_step1', methods=['GET', 'POST'])
def clean_step1():
    global concatenated_df
    # Ensure that concatenated_df is not empty
    if concatenated_df.empty:
        return redirect(url_for('clean'))
    shape_info = None  # Initialize shape information
    shape_info = concatenated_df.shape
    # Count 'inf', '-inf', and NaN values
    num_inf = concatenated_df.isin([np.inf, -np.inf]).sum().sum()
    num_nan = concatenated_df.isna().sum().sum()
    # Count number of duplicate rows
    num_duplicates = concatenated_df.duplicated().sum()
    # Display the concatenated DataFrame
    return render_template('clean_step1.html', 
                           tables=[concatenated_df.head(10).to_html(classes='w3-table-all w3-hoverable')],
                           shape_info=shape_info,
                           num_inf=num_inf, num_nan=num_nan, num_duplicates=num_duplicates)



@app.route('/clean_step2', methods=['GET', 'POST'])
def clean_step2():
    global concatenated_df
    # Ensure that concatenated_df is not empty
    if concatenated_df.empty:
        return redirect(url_for('clean'))
    # Step 2: Remove rows with 'inf', '-inf' and NaN values
    concatenated_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    concatenated_df.dropna(axis=0, how='any', inplace=True)
    # Step 2: Drop duplicate rows
    concatenated_df.drop_duplicates(inplace=True)
    # Count 'inf', '-inf', and NaN values
    num_inf = concatenated_df.isin([np.inf, -np.inf]).sum().sum()
    num_nan = concatenated_df.isna().sum().sum()
    # Count number of duplicate rows
    num_duplicates = concatenated_df.duplicated().sum()
    # Continue with any other data cleaning operations if needed
    shape_info = None  # Initialize shape information
    shape_info = concatenated_df.shape
    return render_template('clean_step2.html', 
                           tables=[concatenated_df.head(10).to_html(classes='w3-table-all w3-hoverable')],
                           shape_info=shape_info,
                           num_inf=num_inf, num_nan=num_nan, num_duplicates=num_duplicates)



@app.route('/clean_step3', methods=['GET', 'POST'])
def clean_step3():
    global concatenated_df
    dropped_columns = []  # Initialize a list to keep track of dropped columns
    # Ensure that concatenated_df is not empty
    if concatenated_df.empty:
        return redirect(url_for('clean'))
    # Step 3: Drop socket features
    socket_features = ['Unnamed: 0', 'Flow ID', ' Source IP', ' Destination IP', ' Source Port', ' Destination Port', 
                       'SimillarHTTP', ' Timestamp']
    # Keep track of dropped columns
    for col in socket_features:
        if col in concatenated_df.columns:
            concatenated_df.drop(columns=[col], inplace=True)
            dropped_columns.append(col)
    # Step 3: Drop features with only one unique value
    for col in concatenated_df.columns:
        if len(concatenated_df[col].unique()) == 1:
            concatenated_df.drop(columns=[col], inplace=True)
            dropped_columns.append(col)
    # Continue with any other data cleaning operations if needed
    shape_info = concatenated_df.shape  # Get updated shape information
    return render_template('clean_step3.html', 
                           tables=[concatenated_df.head(10).to_html(classes='w3-table-all w3-hoverable')],
                           shape_info=shape_info,
                           dropped_columns=dropped_columns)  # Pass dropped columns to the template




@app.route('/clean_step4', methods=['GET', 'POST'])
def clean_step4():
    global concatenated_df
    # Ensure that concatenated_df is not empty
    if concatenated_df.empty:
        return redirect(url_for('clean'))
    if request.method == 'POST':
        clean_filename = request.form.get('clean_filename')
        if not clean_filename:
            return render_template('clean_step4.html', error="Please enter a filename for the cleaned data.")
        # In this step, save the cleaned DataFrame to a clean folder
        clean_folder = 'clean'
        os.makedirs(clean_folder, exist_ok=True)
        clean_file_path = os.path.join(clean_folder, clean_filename + '.csv')
        concatenated_df.to_csv(clean_file_path, index=False)
        return render_template('clean_step4.html', clean_file_path=clean_file_path)
    return render_template('clean_step4.html')



#-----------------------------------------------------------------------------------------------------------
# route to data visualization
app.config['UPLOAD_FOLDER'] = 'upload/'
app.config['CLEAN_FOLDER'] = 'clean/'
@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    if request.method == 'POST':
        folder = request.form['folder']
        return redirect(url_for('visualization_results', folder=folder))
    return render_template('visualization.html')



@app.route('/visualization_results', methods=['GET', 'POST'])
def visualization_results():
    folder = request.args.get('folder')
    label_counts = {}  # Initialize the label_counts dictionary
    plot_base64 = None  # Initialize plot_base64 with None

    # For GET request, initially load the list of available files based on the selected folder
    available_files = []
    if folder == 'upload':
        available_files = [file for file in os.listdir(app.config['UPLOAD_FOLDER']) if file.endswith('.csv')]
    elif folder == 'clean':
        available_files = [file for file in os.listdir(app.config['CLEAN_FOLDER']) if file.endswith('.csv')]
    
    if request.method == 'POST':
        selected_file = request.form['file']
        if folder == 'upload':
            folder_path = app.config['UPLOAD_FOLDER']
        elif folder == 'clean':
            folder_path = app.config['CLEAN_FOLDER']
        if selected_file:
            file_path = os.path.join(folder_path, selected_file)
            df = pd.read_csv(file_path)
            # Perform data visualization tasks here, calculate label counts
            label_counts = df[' Label'].value_counts().to_dict()

            # Create a pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%', startangle=140)
            plt.title('Pie Chart Distribution of Multi-class Labels')
            plt.legend(loc='best')
            plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

            # Save the pie chart to a BytesIO buffer and encode it as base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.read()).decode()

            return render_template('visualization_results.html', folder=folder, available_files=available_files, label_counts=label_counts, plot_base64=plot_base64)

    return render_template('visualization_results.html', folder=folder, available_files=available_files)



#-----------------------------------------------------------------------------------------------------------
# route to data manipulation & transformation
app.config['CLEAN_FOLDER'] = 'clean/'
@app.route('/edition', methods=['GET', 'POST'])
def edition():
    # For GET request, initially load the list of available files based on the selected folder
    available_files = []
    available_files = [file for file in os.listdir(app.config['CLEAN_FOLDER']) if file.endswith('.csv')]
    
    if request.method == 'POST':
        selected_file = request.form.get('file')  # Get the selected file from the form
        selected_action = request.form.get('action')  # Get the selected action from the form
        if selected_file:
            selected_data = pd.read_csv(os.path.join(app.config['CLEAN_FOLDER'], selected_file))
            shape_info = selected_data.shape  # Get the shape information
            if selected_action == 'balance':
                return redirect(url_for('edition_balanced', selected_file = selected_file, shape = shape_info))
            elif selected_action == 'combine':
                return redirect(url_for('edition_combine', selected_file = selected_file, shape = shape_info))
            elif selected_action == 'drop':
                return redirect(url_for('edition_drop', selected_file = selected_file, shape = shape_info))
    return render_template('edition.html', available_files=available_files)



@app.route('/edition_balanced', methods=['GET', 'POST'])
def edition_balanced():
    selected_file = request.args.get('selected_file')
    shape_info = request.args.get('shape')

    # Read the initial dataframe info
    df = pd.read_csv(os.path.join(app.config['CLEAN_FOLDER'], selected_file))
    initial_labels = df[' Label'].value_counts().to_dict()  # Replace with actual labels
    initial_shape = df.shape  # Get the initial shape

    # Define the RandomOverSampler strategy
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

    # Separate features (X) and target labels (y)
    X = df.drop(columns=[' Label'])
    y = df[' Label']

    # Apply random oversampling to balance the classes
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    # Get the labels and counts for the balanced dataset
    balanced_labels = df_resampled[' Label'].value_counts().to_dict()
    balanced_shape = df_resampled.shape

    if request.method == 'POST':
        # User input for saving the balanced dataset
        save_name = request.form.get('save_name')

        # Define the save path based on the folder
        clean_folder = 'clean'
        os.makedirs(clean_folder, exist_ok=True)
        clean_filename = save_name  # Fix variable name here
        clean_file_path = os.path.join(clean_folder, clean_filename + '.csv')
        df_resampled.to_csv(clean_file_path, index=False)

        # Redirect to a success page or the main page after balancing and saving
        return redirect(url_for('edition'))

    return render_template('edition_balanced.html', initial_labels=initial_labels, initial_shape=initial_shape, 
                                                    balanced_labels=balanced_labels, balanced_shape=balanced_shape)


@app.route('/edition_combine', methods=['GET', 'POST'])
def edition_combine():
    selected_file = request.args.get('selected_file')

    # Read the initial dataframe info
    df = pd.read_csv(os.path.join(app.config['CLEAN_FOLDER'], selected_file))
    
    unique_labels = df[' Label'].unique()  # Get unique labels from the dataset
    
    if request.method == 'POST':
        label1 = request.form.get('label1')
        label2 = request.form.get('label2')
        combined_label_name = request.form.get('combined_label_name')  # Get the custom combined label name
        if label1 and label2:
            # Combine the two chosen labels into a new label
            df[' Label'] = df[' Label'].replace([label1, label2], combined_label_name)
            
            # User input for saving the combined dataset
            save_name = request.form.get('save_name')

            # Define the save path based on the folder
            clean_folder = 'clean'
            os.makedirs(clean_folder, exist_ok=True)
            clean_filename = save_name  # Fix variable name here
            clean_file_path = os.path.join(clean_folder, clean_filename + '.csv')
            df.to_csv(clean_file_path, index=False)

            # Redirect to a success page or the main page after combining and saving
            return redirect(url_for('edition'))
    return render_template('edition_combinelabel.html', selected_file=selected_file, unique_labels=unique_labels)



@app.route('/edition_drop', methods=['GET', 'POST'])
def edition_drop():
    selected_file = request.args.get('selected_file')

    # Read the initial dataframe info
    df = pd.read_csv(os.path.join(app.config['CLEAN_FOLDER'], selected_file))
    
    unique_labels = df[' Label'].unique()  # Get unique labels from the dataset
    
    if request.method == 'POST':
        label_to_drop = request.form.get('label_to_drop')
        if label_to_drop:
            # Drop rows with the chosen label
            df = df[df[' Label'] != label_to_drop]
            
            # User input for saving the cleaned dataset
            save_name = request.form.get('save_name')

            # Define the save path based on the folder
            clean_folder = 'clean'
            os.makedirs(clean_folder, exist_ok=True)
            clean_filename = save_name  # Fix variable name here
            clean_file_path = os.path.join(clean_folder, clean_filename + '.csv')
            df.to_csv(clean_file_path, index=False)

            # Redirect to a success page or the main page after dropping and saving
            return redirect(url_for('edition'))

    return render_template('edition_droplabel.html', selected_file=selected_file, unique_labels=unique_labels)
    

#-----------------------------------------------------------------------------------------------------------
# Route for the training page
# Sample algorithm options (you can add more)
algorithm_options = {
}
@app.route('/training', methods=['GET', 'POST'])
def training():
    # Get a list of available CSV files in the 'clean' directory
    dataset_files = [file for file in os.listdir('clean') if file.endswith('.csv')]
    # Check if a file is selected by the user
    selected_file = request.form.get('selected_file')
    selected_data = None
    if selected_file:
        selected_data = pd.read_csv(os.path.join('clean', selected_file))
        # Check if an algorithm is selected
        selected_algorithm = request.form.get('algorithm')
        # Redirect to the training results page
        return redirect(url_for('training_results', selected_file=selected_file, selected_algorithm=selected_algorithm))
    return render_template('training.html', dataset_files=dataset_files, algorithm_options=algorithm_options)



# Define available algorithms and their corresponding functions with parameters
algorithm_options = {
    'Decision Tree': (DecisionTreeClassifier(random_state=42), {}),
    'Random Forest': (RandomForestClassifier(random_state=42), {}),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': 3}),  # You can specify parameters here
    # Add more algorithms as needed
}


@app.route('/training_results', methods=['GET', 'POST'])
def training_results():
    # Get a list of available CSV files in the 'clean' directory
    dataset_files = [file for file in os.listdir('clean') if file.endswith('.csv')]
    
    # Check if a file is selected by the user
    selected_file = request.args.get('selected_file')
    selected_algorithm = request.args.get('selected_algorithm')

    # Load selected data (adjust path as needed)
    data = pd.read_csv(os.path.join('clean', selected_file))

    if selected_algorithm in algorithm_options:
        # Unpack the classifier and parameters
        classifier, params = algorithm_options[selected_algorithm]

        # Separate features and labels
        X = data.drop([' Label'], axis=1)
        y = data[' Label']

        # Perform label encoding on the target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(data[' Label'])

        # Get the original label names
        original_label_names = label_encoder.inverse_transform(np.unique(y_encoded))

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X.values, y_encoded, test_size=0.3, random_state=42)

        # Set the classifier parameters (if any)
        # Note: You can use the classifier and parameters you unpacked earlier
        classifier.set_params(**params)

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = classifier.predict(X_test)

        # Generate a confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create a DataFrame from the confusion matrix
        confusion_matrix_df = pd.DataFrame(cm, columns=original_label_names, index=original_label_names)

        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Compute and format the classification report
        classification_rep = classification_report(y_test, y_pred, target_names=original_label_names, output_dict=True)
        classification_rep_str = classification_report(y_test, y_pred, target_names=original_label_names)

        # Create a heatmap of the confusion matrix with custom labels
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
        xticklabels=original_label_names, yticklabels=original_label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Save the plot to a BytesIO buffer and encode it as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        heatmap_base64 = base64.b64encode(buffer.read()).decode()

        # Save the trained classifier to a file (e.g., trained_model.joblib)
        trained_model_file = 'trained_model.pkl'  # Change the file extension to .pkl
        with open(trained_model_file, 'wb') as model_file:
            pickle.dump(classifier, model_file)

        # Set the global trained_model variable
        global trained_model
        trained_model = trained_model_file

        # Return the results to the template
        return render_template('training_results.html', dataset_files=dataset_files,
                           selected_file=selected_file, selected_algorithm=selected_algorithm,
                           heatmap_filename=heatmap_base64, confusion_matrix_df=confusion_matrix_df,
                           accuracy=accuracy, classification_report=classification_rep_str)

@app.route('/download_model', methods=['GET'])
def download_model():
    global trained_model  # Use the global trained_model variable
    
    # Check if a trained model is available
    if trained_model is not None:
        return send_file(trained_model, as_attachment=True)
    else:
        return "Trained model not available."



#-----------------------------------------------------------------------------------------------------------
@app.route('/dashboard_files', methods=['GET', 'POST'])
def dashboard_files():
    if 'user_id' not in session:
        return redirect(url_for('input'))
    # Get the list of uploaded files and cleaned files
    upload_files = os.listdir(app.config['UPLOAD_FOLDER'])
    clean_files = os.listdir(app.config['CLEAN_FOLDER'])
                                                                            # https://www.educative.io/answers/how-to-download-files-in-flask https://flexiple.com/python/python-remove-file
    if request.method == 'POST':
        if 'download_files' in request.form:
            selected_files = request.form.getlist('multiple_files_selection[]')
            for file_name in selected_files:
                # Implement code to download each selected file
                file_path = None
                if file_name in upload_files:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                elif file_name in clean_files:
                    file_path = os.path.join(app.config['CLEAN_FOLDER'], file_name)
                if file_path and os.path.exists(file_path):
                    # Download the file
                    return send_file(file_path, as_attachment=True)
                else:
                    flash(f'File not found: {file_name}', 'error')

        elif 'delete_files' in request.form:
            files_to_delete = request.form.getlist('multiple_files_selection[]')
            for file_name in files_to_delete:
                # Check if the file exists in the upload folder or clean folder
                upload_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
                clean_file_path = os.path.join(app.config['CLEAN_FOLDER'], file_name)
                if os.path.exists(upload_file_path):
                    os.remove(upload_file_path)
                elif os.path.exists(clean_file_path):
                    os.remove(clean_file_path)
                else:
                    flash(f'File not found: {file_name}', 'error')

            # After deleting all selected files, redirect to the 'dashboard_files' page
            return redirect(url_for('dashboard_files'))

    return render_template('dashboard_files.html', upload_files=upload_files, clean_files=clean_files)

  

    

from datetime import datetime  # Updated import statement
model = None  # Initialize the model
loaded_data = None  # Initialize the data for monitoring
label_encoder = None  # Initialize the label encoder
import tempfile

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('input'))
    global model, loaded_data, label_encoder  # Include label_encoder in the global scope
    if request.method == 'POST':
        upload_csv = request.files['csv_file']
        upload_model = request.files['model_file']
        upload_encoder = request.files['encoder_file']  # Add input for Label Encoder file

          # Check if files are uploaded
        if not upload_csv or not upload_model or not upload_encoder:
            msg = 'Please upload CSV, model, and encoder files.'
            return render_template('dashboard.html', msg=msg)  # Render the message in the template

        # Check file types
        if not upload_csv.filename.endswith('.csv'):
            msg = 'Invalid CSV file format. Please upload a CSV file.'
            return render_template('dashboard.html', msg=msg)  # Render the message in the template

        if not upload_model.filename.endswith('.pkl'):
            msg = 'Invalid model file format. Please upload a pickle (.pkl) model file.'
            return render_template('dashboard.html', msg=msg)  # Render the message in the template

        if not upload_encoder.filename.endswith('.csv'):
            msg = 'Invalid encoder file format. Please upload a CSV encoder file.'
            return render_template('dashboard.html', msg=msg)  # Render the message in the template

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        # Save the uploaded model to the temporary directory
        temp_model_path = os.path.join(temp_dir, upload_model.filename)
        upload_model.save(temp_model_path)

        # Load the uploaded CSV file
        df = pd.read_csv(upload_csv)
        encode = pd.read_csv(upload_encoder)

        # Perform any preprocessing on the CSV data if required
        label_encoder = LabelEncoder()
        encoded = label_encoder.fit_transform(encode[' Label'])
        if ' Label' in df.columns:
            df.drop([' Label'], axis=1, inplace=True)
        
# https://practicaldatascience.co.uk/machine-learning/how-to-save-and-load-machine-learning-models-using-pickle#:~:text=To%20load%20a%20saved%20model,back%20an%20array%20of%20predictions. 

        # Load the machine learning model using pickle
        with open(temp_model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Use the loaded model to predict labels for the uploaded data
        predictions = model.predict(df)  # Use the loaded model for predictions

        # Inverse transform the encoded predictions to get original labels
        predictions_label = label_encoder.inverse_transform(predictions)
        # Add the predictions to the DataFrame
        df['Predicted_Label'] = predictions_label

        # Create a time and date column
        df['Time'] = datetime.now().strftime("%H:%M:%S")
        df['Date'] = datetime.now().strftime("%Y-%m-%d")

        # Update the loaded_data with the new data
        loaded_data = df

        # Clean up: Remove the temporary directory and its contents
        shutil.rmtree(temp_dir)

        return render_template('dashboard.html', data=loaded_data.to_dict(orient='records'))

    return render_template('dashboard.html')


# Disable caching for API endpoints
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Run the Flask application if this script is executed as the main program
if __name__ == '__main__':
    run = True 
    app.run(host='0.0.0.0', debug=False, threaded=True, port=5000)