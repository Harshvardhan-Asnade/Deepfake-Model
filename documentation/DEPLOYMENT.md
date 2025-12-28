# DeepGuard Deployment Guide

This guide covers how to deploy the **Frontend** to **Vercel** and the **Backend** to **Hugging Face Spaces**.

---

## 1. Frontend Deployment (Vercel)

The frontend is a static site (HTML/CSS/JS) and can be hosted for free on Vercel.

### Prerequisites
- A [Vercel account](https://vercel.com).
- GitHub repository with your project code.

### Configuration
We have already created a `vercel.json` file in the `frontend` folder to handle routing:

```json
{
  "version": 2,
  "builds": [
    { "src": "*.html", "use": "@vercel/static" }
  ],
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

### Steps
1.  **Push your code** to GitHub.
2.  Go to the [Vercel Dashboard](https://vercel.com/dashboard) and click **"Add New Project"**.
3.  Import your GitHub repository.
4.  **Configure Project Settings**:
    - **Framework Preset**: select "Other".
    - **Root Directory**: Click "Edit" and select `frontend`.
5.  Click **Deploy**.

Your frontend will be live at `https://your-project.vercel.app`.

---

## 2. Backend Deployment (Hugging Face Spaces)

The backend requires a Python environment with PyTorch and GPU support, making **Hugging Face Spaces** an ideal free/cheap host.

### Prerequisites
- A [Hugging Face account](https://huggingface.co).

### Steps to Deploy on HF Spaces

1.  **Create a New Space**:
    - Go to [huggingface.co/spaces](https://huggingface.co/spaces).
    - Click **"Create new Space"**.
    - **Name**: `deepguard-backend`.
    - **SDK**: Select **Docker**.
    - **Hardware**: Select **CPU Basic** (free) or upgrade to **T4 GPU** (recommended for speed).

2.  **Upload Backend Code**:
    - Clone your new Space locally:
      ```bash
      git clone https://huggingface.co/spaces/YOUR_USERNAME/deepguard-backend
      ```
    - Copy the contents of your `backend` folder into this new directory.
    - Copy your `model` folder into the root of the Space as well.

    _The `Dockerfile` and `app.py` configuration have already been prepared for you in the `huggingface_deploy` folder._

4.  **Update `app.py` for Hugging Face**:
    Change the run command at the bottom of `app.py`:
    ```python
    app.run(host='0.0.0.0', port=7860)
    ```

5.  **Push and Build**:
    ```bash
    git add .
    git commit -m "Deploy backend"
    git push
    ```
    Hugging Face will automatically build your Docker container. Once "Running", your backend API will be available at `https://YOUR_USERNAME-deepguard-backend.hf.space`.

---

## 3. Connect Frontend to Backend

Once your backend is live:

1.  Open `frontend/script.js` in your local project.
2.  Find the `API_BASE_URL` configuration at the top of the file:

    ```javascript
    const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? '' 
        : 'https://YOUR-HUGGINGFACE-SPACE-URL.hf.space';
    ```

3.  Replace the placeholder URL with your actual Hugging Face Space URL.
4.  Commit and push the changes to GitHub.
5.  Vercel will automatically redeploy your frontend with the new configuration.

---

## Troubleshooting

-   **CORS Issues**: If fetch requests fail, ensure `flask-cors` is enabled in `app.py` (it is by default).
-   **Model Loading Errors**: Ensure the `model/` directory is correctly copied to the Hugging Face Space and the path in `config.py` is correct relative to the Docker container structure.
