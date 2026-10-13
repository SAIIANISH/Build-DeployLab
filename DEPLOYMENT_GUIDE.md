# Deployment Guide
## Electric Vehicle Range Prediction ML Web Application

This guide explains how to run the application in a production-ready mode locally and how to deploy it to free cloud hosting platforms (such as Render or PythonAnywhere).

---

## 1. Running in Production Locally

By default, `python app.py` runs Flask's built-in development server. For production grade concurrency:

### On Windows (Using Waitress WSGI Server)
1. Install Waitress:
   ```powershell
   pip install waitress
   ```
2. Start the production server:
   ```powershell
   waitress-serve --port=7000 app:app
   ```

### On Linux / macOS (Using Gunicorn)
1. Install Gunicorn:
   ```bash
   pip install gunicorn
   ```
2. Start the server with 4 worker processes:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:7000 app:app
   ```

---

## 2. Deploying to Cloud Platforms

### Option A: Render.com (Recommended & Free)
1. **Push your code to GitHub:**
   - Initialize git: `git init`
   - Add files: `git add .`
   - Commit: `git commit -m "Initial EV Range ML commit"`
   - Push to your GitHub repository.

2. **Create a Web Service on Render:**
   - Log into [Render.com](https://render.com).
   - Click **New +** > **Web Service**.
   - Connect your GitHub repository.

3. **Configure Settings:**
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app` (or `python app.py`)

4. Click **Deploy Web Service**. Your app will be live with a free HTTPS `.onrender.com` URL!

---

### Option B: PythonAnywhere (Free & Beginner Friendly)
1. Create a free account on [PythonAnywhere.com](https://www.pythonanywhere.com).
2. Go to the **Consoles** tab and open a Bash console.
3. Clone your GitHub repository or upload your project folder.
4. Set up a virtual environment and install packages:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.10 ev-env
   pip install -r requirements.txt
   python train_model.py
   ```
5. Go to the **Web** tab, click **Add a new web app**, choose **Manual Configuration (Python 3.10)**.
6. Under the **Code** section:
   - Set **Source code:** to your project directory.
   - Edit the **WSGI configuration file** to point to your `app`:
     ```python
     import sys
     path = '/home/yourusername/EV_Range_Prediction'
     if path not in sys.path:
         sys.path.insert(0, path)
     from app import app as application
     ```
7. Click **Reload**. Your site is now live at `yourusername.pythonanywhere.com`.
