import pyttsx3

def test_voices():
    # Initialize the engine
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    # Test text in both languages
    test_texts = {
        "English": "Hello, this is a test of the English voice.",
        "Spanish": "Hola, esta es una prueba de la voz en español."
    }
    
    print("\n=== Available Voices ===")
    for idx, voice in enumerate(voices):
        print(f"\nVoice #{idx}")
        print(f"ID: {voice.id}")
        print(f"Name: {voice.name}")
        print(f"Languages: {voice.languages}")
        print(f"Gender: {voice.gender}")
        print(f"Age: {voice.age}")
        
        # Set the current voice
        engine.setProperty('voice', voice.id)
        
        # Ask if user wants to test this voice
        response = input("\nWould you like to test this voice? (y/n): ")
        if response.lower() == 'y':
            print("\nTesting voice in English...")
            engine.say(test_texts["English"])
            engine.runAndWait()
            
            print("\nTesting voice in Spanish...")
            engine.say(test_texts["Spanish"])
            engine.runAndWait()
            
            # Ask for rating
            rating = input("\nHow would you rate this voice (1-5)? ")
            print(f"You rated voice {voice.name} as {rating}/5")
        
        print("\n" + "="*50)
        
        # Ask if user wants to continue
        if idx < len(voices) - 1:  # Don't ask after last voice
            cont = input("\nContinue to next voice? (y/n): ")
            if cont.lower() != 'y':
                break

    print("\nVoice testing complete!")
    print("\nFor reference, here are all the voices you rated highly:")
    for idx, voice in enumerate(voices):
        print(f"{idx}: {voice.id} - {voice.name}")

if __name__ == "__main__":
    test_voices()