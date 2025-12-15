# 🔴 Live Demo Guide (Tech Expo 2026)

**"Nothing beats a live demo."**
Follow this checklist to ensure your presentation goes smoothly and you impress the judges with a working prototype.

## 1️⃣ Preparation (5 Minutes Before)

*   [ ] **Clean Your Desktop**: Close unnecessary tabs and apps.
*   [ ] **Start the Server**:
    *   Open Terminal.
    *   Run: `python app.py`
    *   *Keep this terminal window open but minimized.*
*   [ ] **Pre-load the Web Page**:
    *   Open Chrome.
    *   Go to `http://localhost:5001`.
    *   *Zoom in to 110% so the audience can see clearly.*
*   [ ] **Prepare Test Images**:
    *   Have a folder named `Demo_Images` on your Desktop.
    *   Include:
        *   `1_Real_Person.jpg` (A clean, real photo)
        *   `2_Deepfake.jpg` (A known fake that your model detects well)

## 2️⃣ The Script (What to Say & Do)

**Step 1: The "Real" Test**
*   **Say**: "Let's first test the system with a real photograph."
*   **Do**: Drag and drop `1_Real_Person.jpg` into the upload box.
*   **Wait**: (1-2 seconds)
*   **Show**: Point to the "REAL" result and the clean heatmap (mostly blue).
*   **Say**: "As you can see, the model correctly identifies natural artifacts."

**Step 2: The "Deepfake" Reveal**
*   **Say**: "Now, let's try a state-of-the-art AI generated image."
*   **Do**: Drag and drop `2_Deepfake.jpg`.
*   **Wait**: (1-2 seconds)
*   **Show**: Point to the "FAKE" result and the **Red** spots on the heatmap.
*   **Say**: "DeepGuard instantly flags the inconsistencies in the hair and background."

## 3️⃣ Troubleshooting (Panic Button) 🆘

**If the Demo Crashes:**
1.  **Don't Panic.**
2.  Switch to your **Backup Slides**.
3.  **Say**: "It seems the demo gods aren't with us today, but here are the results from our testing bench."
4.  Show the `presentation_assets/3_heatmap_overlay.jpg` image.

**If "Model Not Found":**
*   Ensure you ran `python app.py` from the `Morden Detections system` folder, not inside `src` or `frontend`.

---

**Good Luck! You got this. 🚀**
