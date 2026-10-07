Cardio Detection App - Ready-to-Run
----------------------------------

What is included:
- train_model.py   : trains the ANN and saves model to model/ann_model.h5
- predict.py       : prediction helper that loads model and scaler
- app.py           : simple Flask web app (upload image -> predict)
- templates/index.html : minimal web UI
- datasets/        : place your ECG images here (four folders already created)
    - normal
    - mi
    - hmi
    - haveingabnormalheartbeat
- model/           : will contain saved model after training
- uploads/         : uploaded images are stored here when using the web app
- requirements.txt : list of Python packages

Steps to run (Windows):
1. Unzip cardio_detection_app.zip
2. Move your ECG image files into the datasets subfolders. Keep the folder names exactly:
   normal, mi, hmi, haveingabnormalheartbeat
   Ensure images are .jpg, .png or .jpeg
3. Create and activate a virtual environment (recommended):
     python -m venv venv
     venv\\Scripts\\activate
4. Install requirements:
     pip install -r requirements.txt
5. Train the model (this will create model/ann_model.h5):
     python train_model.py
6. Run the web app:
     python app.py
   Open http://127.0.0.1:5000 in your browser and upload an ECG image to predict.

Notes:
- Keep it simple: training prints accuracy to the terminal.
- If you have few images, the model may overfit — collect more labeled images for better results.
- If TensorFlow installation fails on Python 3.13 on your machine, consider using Python 3.10 or 3.11.