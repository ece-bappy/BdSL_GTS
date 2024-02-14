import pyttsx3

# Initialize the TTS engine
engine = pyttsx3.init()

# Input the text you want to convert to speech
text_to_speak = "amar sonar bangla, ami tomay valobashi"

# Set the properties for the TTS engine (optional)
engine.setProperty("rate", 150)  # Speed of speech (words per minute)
engine.setProperty("volume", 1.0)  # Volume (0.0 to 1.0)

# Convert text to speech
engine.say(text_to_speak)

# Wait for the speech to finish
engine.runAndWait()
