import pyttsx3


def bangla_to_speech(text):
    engine = pyttsx3.init()

    # On some systems, you might need to specify the voice ID for Bangla.
    # For this, you might have to explore the available voices and choose the appropriate one.
    voices = engine.getProperty("voices")
    for voice in voices:
        if "bangla" in voice.name.lower() or "bengali" in voice.name.lower():
            engine.setProperty("voice", voice.id)
            break

    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    text = " আমি বাংলায় গান গাই"
    bangla_to_speech(text)
