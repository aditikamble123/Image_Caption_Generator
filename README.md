## ✨ Caption Recommender
Upload a photo and get aesthetic Instagram captions using AI!

## 📝 Overview
The Caption Recommender is an AI-powered web app that suggests Instagram captions based on the visual content of your image.

Built using:
OpenAI's CLIP model for image understanding
Streamlit for an interactive and beautiful web UI

## 🎁 Features
✅ Upload an image (jpg, png, jpeg)
✅ Automatically recommends aesthetic captions based on the image content
✅ Clean, modern dark mode UI
✅ Captions displayed in beautiful pink cards
✅ Super simple to use — no coding required
✅ Powered by CLIP + cosine similarity + pre-loaded captions CSV

## 🚀 How It Works

You upload an image

The image is passed through CLIP to generate an image embedding

The app compares this embedding with text embeddings of aesthetic captions

The top 5 most similar captions are shown to you — ready to copy & paste!

## 🛠️ Tech Stack
Streamlit — Web app framework

CLIP — Vision-language model

Python libraries:

torch

transformers

Pandas

PIL

Streamlit

