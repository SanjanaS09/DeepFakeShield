# from google import genai
# from PIL import Image

# # 🔐 SET API KEY
# client = genai.Client(
#     api_key="")

# # ==========================================
# # 🖼️ IMAGE REASONING
# # ==========================================
# def generate_reasoning_with_image(image_path, detection_result, features):
#     try:
#         prompt = f"""
#         You are an expert in deepfake detection.

#         Model Prediction: {detection_result.get("prediction")}
#         Confidence: {detection_result.get("confidence")}

#         Model Signals:
#         {features.get("reasoning", [])}

#         Analyze this image and explain:
#         1. Why it is REAL or FAKE
#         2. Where manipulation artifacts are visible
#         3. Correlate model signals with visual evidence

#         Keep explanation simple and human-friendly.
#         """

#         img = Image.open(image_path).convert("RGB")

#         response = client.models.generate_content(
#             model="gemini-1.5-flash",
#             contents=[prompt, img]
#         )

#         return response.text

#     except Exception as e:
#         print("Gemini image error:", e)
#         return fallback_text(detection_result)


# # ==========================================
# # 🎥 VIDEO REASONING (FRAMES)
# # ==========================================
# def generate_reasoning_with_video(frame_paths, detection_result, features):
#     try:
#         frames = []

#         for path in frame_paths[:3]:
#             img = Image.open(path).convert("RGB")
#             frames.append(img)

#         prompt = f"""
#         You are an expert in deepfake detection.

#         Model Prediction: {detection_result.get("prediction")}
#         Confidence: {detection_result.get("confidence")}

#         Temporal Signals:
#         {features.get("reasoning", [])}

#         Analyze these frames and explain:
#         - Motion inconsistencies
#         - Facial mismatches
#         - Temporal artifacts

#         Keep explanation simple.
#         """

#         response = client.models.generate_content(
#             model="gemini-1.5-flash",
#             contents=[prompt] + frames
#         )

#         return response.text

#     except Exception as e:
#         print("Gemini video error:", e)
#         return fallback_text(detection_result)


# # ==========================================
# # 🔊 AUDIO REASONING
# # ==========================================
# def generate_reasoning_audio(audio_path, detection_result, features):
#     try:
#         prompt = f"""
#         You are an expert in deepfake audio detection.

#         Model Prediction: {detection_result.get("prediction")}
#         Confidence: {detection_result.get("confidence")}

#         Audio Signals:
#         {features.get("reasoning", [])}

#         Analyze this audio and explain:
#         - Voice inconsistencies
#         - Synthetic patterns
#         - Why it is fake or real

#         Keep explanation simple.
#         """

#         with open(audio_path, "rb") as f:
#             audio_bytes = f.read()

#         response = client.models.generate_content(
#             model="gemini-1.5-flash",
#             contents=[
#                 prompt,
#                 {
#                     "mime_type": "audio/wav",
#                     "data": audio_bytes
#                 }
#             ]
#         )

#         return response.text

#     except Exception as e:
#         print("Gemini audio error:", e)
#         return fallback_text(detection_result)


# # ==========================================
# # 🛟 FALLBACK (IMPORTANT)
# # ==========================================
# def fallback_text(result):
#     if result.get("prediction") == "FAKE":
#         return (
#             "The media shows signs of manipulation such as inconsistencies "
#             "in facial structure, unnatural textures, or irregular patterns."
#         )
#     else:
#         return (
#             "The media appears authentic with consistent features and no visible signs of manipulation."
#         )

