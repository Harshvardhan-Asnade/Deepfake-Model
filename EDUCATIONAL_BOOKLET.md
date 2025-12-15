# 📘 Educational Booklet: Understanding Deepfakes

## What is a Deepfake?
"Deepfake" is a portmanteau of "Deep Learning" and "Fake". It refers to synthetic media in which a person in an existing image or video is replaced with someone else's likeness, or purely generated from scratch using Artificial Intelligence.

## How are they created?
Modern deepfakes are generated using two main technologies:

1.  **GANs (Generative Adversarial Networks)**:
    *   Two AIs fight each other. One (the **Generator**) tries to create a fake image. The other (the **Discriminator**) tries to detect it.
    *   Over time, the Generator gets so good that the Discriminator can't tell the difference.
    *   *Examples*: StyleGAN, FaceSwap.

2.  **Diffusion Models**:
    *   The model learns by destroying an image with noise (static) until it is unrecognizable, and then learning to reverse the process to reconstruct the clear image.
    *   It can then start with pure noise and "hallucinate" a completely new image.
    *   *Examples*: Stable Diffusion, Midjourney, DALL-E.

## Why are they dangerous?
*   **Misinformation**: Creating fake news footage of politicians saying things they never said.
*   **Fraud**: Impersonating CEOs to authorize bank transfers.
*   **Harassment**: Non-consensual deepfake pornography.
*   **Erosion of Trust**: If we can't believe our eyes, we may stop believing authentic footage too (the "Liar's Dividend").

## How do we catch them? (Forensics 101)
Even advanced AI leaves traces:
*   **The Grid Effect**: AI generates images in blocks (checkboards). While invisible to the eye, computers can see widely spaced "grid" artifacts in the frequency spectrum.
*   **Symmetry Issues**: AI often struggles with earrings, eyeglasses, or iris reflections matching perfectly.
*   **Text & Hands**: While improving, AI often garbles background text or gives people 6 fingers.
*   **Lighting Physics**: AI is "painting" pixels, not simulating light. Often, shadows will point in different directions.

## Our Solution
**DeepGuard** doesn't rely on just one clue. By looking at the image spatially (RGB), spectrally (Frequency), and locally (Patches), we aim to catch defects that the generator missed.
