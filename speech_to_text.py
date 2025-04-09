import azure.cognitiveservices.speech as speechsdk
import requests
import io
import re
import base64
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor

# Azure Speech Service credentials
speech_key = "25h4UbO95jsfP0TZ2urK0akwXeH4CHcCBgHiFiLQkNaKA858YBBTJQQJ99BDACGhslBXJ3w3AAAYACOGk1cL"
service_region = "centralindia"

# Together API Key
together_api_key = "4be69497833ef67be1e7425672b9132ad1ff9ceee0a1903549abf59b0d367ba2"

def recognize_from_base64(base64_audio):
    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(base64_audio)
        print(f"Audio data received: {len(audio_bytes)} bytes")
        
        # Convert WebM to WAV using pydub
        from pydub import AudioSegment
        import tempfile
        import os
        
        # Save original audio to temp file
        webm_path = tempfile.NamedTemporaryFile(suffix='.webm', delete=False).name
        with open(webm_path, 'wb') as f:
            f.write(audio_bytes)
        
        try:
            # Convert to WAV
            audio = AudioSegment.from_file(webm_path)
            wav_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            audio.export(wav_path, format="wav")
            
            # Use file-based audio config with the converted WAV
            audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
            
            # Speech recognizer setup
            speech_config = speechsdk.SpeechConfig(
                subscription="4be69497833ef67be1e7425672b9132ad1ff9ceee0a1903549abf59b0d367ba2",
                region="centralindia"
            )
            speech_config.speech_recognition_language = "en-US"  # Default to English
            
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            print("🧠 Recognizing speech...")
            result = speech_recognizer.recognize_once_async().get()
            
            # Clean up temp files
            os.remove(webm_path)
            os.remove(wav_path)
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                print(f"Recognition successful: '{result.text}'")
                return result.text
            else:
                print(f"Recognition failed with reason: {result.reason}")
                return "Speech not recognized"
                
        except Exception as conversion_error:
            print(f"Audio conversion error: {str(conversion_error)}")
            return f"Audio conversion error: {str(conversion_error)}"
            
    except Exception as e:
        print(f"Error in speech recognition: {str(e)}")
        return f"Error: {str(e)}"
# Configuration
class Config:
    # Speech recognition timeouts
    END_SILENCE_TIMEOUT = "500"  # ms
    INITIAL_SILENCE_TIMEOUT = "1500"  # ms
    
    # Thread pool for concurrent operations
    MAX_WORKERS = 5
    
    # Response caching
    ENABLE_RESPONSE_CACHE = True
    
    # Voice settings
    ENGLISH_VOICE = "en-IN-NeerjaNeural"  # Female voice
    HINDI_VOICE = "hi-IN-SwaraNeural"     # Female voice
    
    # Response times
    LLM_TIMEOUT = 3  # seconds
    
    # Language detection
    MIN_CONFIDENCE_THRESHOLD = 0.4
    DEFAULT_LANGUAGE = "en-IN"

# Global state manager
class State:
    def __init__(self):
        self.current_language = Config.DEFAULT_LANGUAGE
        self.is_synthesizing = False
        self.is_processing = False
        self.introduction_played = False
        self.first_interaction = True
        self.messages = []
        self.speech_recognizer = None
        self.executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
        self.last_detected_language = None
        self.recognition_active = False
        self.speech_config = None
        self.audio_config = None
        self.language_switch_requested = False
        self.language_recognizers = {}  # Cache for language-specific recognizers
        self.last_recognition_time = time.time()
        self.recognizer_lock = threading.Lock()  # Lock for thread safety
        self.switching_language = False  # Flag to track language switching process
        
    def reset_processing_flags(self):
        self.is_processing = False
        self.switching_language = False

# Initialize state
state = State()

# Setup Azure Speech Config
def setup_speech_config():
    state.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    state.audio_config = speechsdk.AudioConfig(use_default_microphone=True)
    
    # Optimize for conversational speech
    state.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, Config.END_SILENCE_TIMEOUT)
    state.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, Config.INITIAL_SILENCE_TIMEOUT)
    
    # Configure for cleaner response
    state.speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "")  # Disable logging
    
    # Set initial language
    state.speech_config.speech_recognition_language = state.current_language

# Response cache - common phrases
response_cache = {
    "hello": {
        "en-IN": "TAG_ENGLISH Hello! How can I assist you with your Air India flight today?",
        "hi-IN": "TAG_HINDI नमस्ते! मैं आज आपकी एयर इंडिया उड़ान के साथ कैसे मदद कर सकता हूँ?"
    },
    "hi": {
        "en-IN": "TAG_ENGLISH Hi there! How can I help you with your Air India travel needs?",
        "hi-IN": "TAG_HINDI नमस्कार! मैं आपकी एयर इंडिया यात्रा की ज़रूरतों के साथ कैसे मदद कर सकता हूँ?"
    },
    "check pnr": {
        "en-IN": "TAG_ENGLISH To check your PNR status, I'll need your 6-digit PNR number. Please say it now.",
        "hi-IN": "TAG_HINDI आपके पीएनआर स्थिति की जांच करने के लिए, मुझे आपके 6-अंकीय पीएनआर नंबर की आवश्यकता होगी। कृपया अब बताएं।"
    },
    "pnr status": {
        "en-IN": "TAG_ENGLISH To check your PNR status, I'll need your 6-digit PNR number. Please say it now.",
        "hi-IN": "TAG_HINDI आपके पीएनआर स्थिति की जांच करने के लिए, मुझे आपके 6-अंकीय पीएनआर नंबर की आवश्यकता होगी। कृपया अब बताएं।"
    },
    "switch to hindi": {
        "hi-IN": "TAG_HINDI हिंदी में बात करने के लिए तैयार हूँ।"
    },
    "switch to english": {
        "en-IN": "TAG_ENGLISH I've switched to English now."
    },
    # Add PNR related cache entries
    "pnr": {
        "en-IN": "TAG_ENGLISH To check your PNR status, please provide your 6-digit PNR number.",
        "hi-IN": "TAG_HINDI अपने पीएनआर स्थिति की जांच करने के लिए, कृपया अपना 6-अंकीय पीएनआर नंबर बताएं।"
    }
}

# Introduction messages for startup
introduction_messages = {
    "en-IN": "TAG_ENGLISH Hello! I'm Air India's voice assistant. How may I assist you today?",
    "hi-IN": "TAG_HINDI नमस्ते! मैं एयर इंडिया का वॉयस असिस्टेंट हूं। आज मैं आपकी कैसे सहायता कर सकता हूं?"
}

# Enhanced system prompt with stronger language switching instruction
system_prompt = """You are Air India's friendly and helpful voice assistant that speaks both Hindi and English.

CRITICAL LANGUAGE RULES:
- IMMEDIATELY detect and switch to the language of the user input
- Hindi input (including transliterated Hindi) MUST receive Hindi output with TAG_HINDI prefix.
- English input MUST receive English output with TAG_ENGLISH prefix.
- NEVER mix languages in a single response.
- Keep responses brief and conversational (under 20 words when possible).
- NEVER add language introductions or acknowledgments in responses.

FUNCTIONALITY:
- For PNR status inquiries: Ask for the 6-digit PNR number if not provided
- When PNR is detected: Respond with CALL_PNR_API <PNR>
- For name changes: Collect PNR, old name, new name, then respond with: PROCESS_NAME_CHANGE <PNR>;<OLD_NAME>;<NEW_NAME>
- For flight information: Collect details and provide relevant information
- For other common issues (baggage, meals, etc.): Provide helpful information

Remember to keep your tone friendly, professional and efficient like a major airline's customer service.
"""

# PNR pattern for direct recognition
pnr_pattern = re.compile(r'\b(\d{6})\b')

# Improved language detection
def detect_language(text):
    if not text or len(text.strip()) == 0:
        return state.current_language, 0.9  # Default to current if empty
    
    text = text.strip().lower()
    
    # English switch commands
    if any(phrase in text for phrase in ["speak in english", "switch to english", "talk in english", 
                                       "english please", "english mode", "english language", 
                                       "change to english", "reply in english", "speak english"]):
        state.language_switch_requested = True
        return "en-IN", 1.0
    
    # Hindi switch commands - comprehensive list including transliterated versions    
    if any(phrase in text for phrase in ["hindi me", "hindi mein", "hindi mai", "hindi में", "हिंदी में", 
                                       "हिंदी में बात करो", "हिंदी में बताओ", "हिंदी में जवाब दो", 
                                       "speak in hindi", "switch to hindi", "hindi please", 
                                       "hindi mode", "change to hindi", "talk in hindi", 
                                       "baat karo hindi me", "hindi language"]):
        state.language_switch_requested = True
        return "hi-IN", 1.0
    
    # Check for Hindi characters
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    if hindi_chars > 0:
        return "hi-IN", 0.95
    
    # Common Hindi words (including transliterated)
    hindi_indicators = [
        "namaste", "namaskar", "kripya", "dhanyawad", "shukriya", "mujhe", "hume", "humko", 
        "chahiye", "hai", "hain", "aur", "kya", "kaun", "kahan", "kaise", "kyun", "kab",
        "apka", "tumhara", "ki", "ka", "mera", "humara", "apna", "jankari", "sthiti", 
        "batao", "bataye", "suniye", "suno", "karo", "kariye", "ticket", "jaankari",
        "khana", "khane", "samay", "time", "seat", "name", "mai", "mein", "se", "ko"
    ]
    
    # Common English airline words
    english_indicators = [
        "hello", "hi", "please", "thank", "thanks", "want", "need", "check", "status", "flight",
        "booking", "cancel", "change", "help", "assist", "air", "india", "pnr", "ticket",
        "my", "what", "how", "when", "where", "why", "tell", "me", "about", "can", "you",
        "airport", "baggage", "luggage", "meal", "vegetarian", "seat", "time", "departure",
        "arrival", "delay", "refund", "name", "passenger", "confirmation", "number"
    ]
    
    # Count word matches
    hindi_count = sum(1 for word in hindi_indicators if re.search(r'\b' + word + r'\b', text))
    english_count = sum(1 for word in english_indicators if re.search(r'\b' + word + r'\b', text))
    
    # Weighted scoring
    hindi_weighted = hindi_count * 1.5
    english_weighted = english_count * 1.0
    
    total_weighted = hindi_weighted + english_weighted
    
    # If we have a clear signal
    if total_weighted > 0:
        if hindi_weighted > 0 and english_weighted == 0:
            return "hi-IN", min(0.9 + (hindi_count * 0.02), 1.0)
        elif english_weighted > 0 and hindi_weighted == 0:
            return "en-IN", min(0.8 + (english_count * 0.02), 1.0)
        
        # Mixed language - determine dominant with hindi bias
        hindi_ratio = hindi_weighted / total_weighted
        if hindi_ratio >= 0.4:  # Lower threshold to favor Hindi
            return "hi-IN", hindi_ratio + 0.1  # Boost confidence
        else:
            return "en-IN", (1 - hindi_ratio) + 0.1  # Boost confidence
    
    # Short inputs & numbers - maintain context
    if len(text.split()) <= 2 or re.search(r'^\d+$', text.strip()):
        return state.current_language, 0.7
    
    # Default to current language
    return state.current_language, 0.6

# Speech recognizer factory with caching
def create_speech_recognizer(language_code):
    # Check if we already have a cached recognizer for this language
    if language_code in state.language_recognizers and state.language_recognizers[language_code] is not None:
        print(f"Using cached recognizer for: {language_code}")
        return state.language_recognizers[language_code]
    
    print(f"Creating new recognizer for: {language_code}")
    
    # Create a new speech config for this language
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = language_code
    
    # Set timeout properties
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, 
        Config.END_SILENCE_TIMEOUT
    )
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
        Config.INITIAL_SILENCE_TIMEOUT
    )
    
    # Create the recognizer
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        audio_config=state.audio_config
    )
    
    # Cache the recognizer
    state.language_recognizers[language_code] = recognizer
    return recognizer

# PNR details function - mocked for testing
def get_pnr_details(pnr_number):
    # Mock data for immediate response
    return {
        "pnr_number": pnr_number,
        "pnr_status": "Confirmed",
        "arrival_time": "10:30 AM",
        "departure_time": "8:00 AM",
        "flight_no": "AI-123",
        "passenger_name": "Sharma",
        "seat": "14A",
        "date": "March 22, 2025"
    }

# Name change processor
def process_name_change(pnr_number, old_name, new_name):
    print(f"\n----- Processing Name Change Request -----")
    print(f"PNR: {pnr_number}")
    print(f"Old Name: {old_name}")
    print(f"New Name: {new_name}")
    print(f"-----------------------------------------\n")
    return True

# Check for PNR related keywords
def is_pnr_query(text):
    text = text.lower()
    # Check for PNR related keywords in English
    english_pnr_keywords = ["pnr", "status", "booking", "reference", "number", "ticket"]
    # Check for PNR related keywords in Hindi (including transliterated)
    hindi_pnr_keywords = ["pnr", "पीएनआर", "स्थिति", "स्टेटस", "बुकिंग", "टिकट", "नंबर", "jankari", "status", "number"]
    
    # Check for any PNR related keywords
    return any(keyword in text for keyword in english_pnr_keywords + hindi_pnr_keywords)

# Optimized LLM response
def get_llm_response(prompt, detected_language):
    # Fast path: Check for PNR number pattern
    pnr_match = pnr_pattern.search(prompt)
    if pnr_match:
        pnr_number = pnr_match.group(1)
        return handle_pnr_response(pnr_number, detected_language)
    
    # Fast path: Check for PNR related query
    if is_pnr_query(prompt):
        if detected_language == "hi-IN":
            return "TAG_HINDI अपने पीएनआर स्थिति की जांच करने के लिए, कृपया अपना 6-अंकीय पीएनआर नंबर बताएं।"
        else:
            return "TAG_ENGLISH To check your PNR status, please provide your 6-digit PNR number."
    
    # Fast path: Check for language switch commands
    if state.language_switch_requested:
        if detected_language == "hi-IN":
            return "TAG_HINDI हिंदी में बात करने के लिए तैयार हूँ। कैसे मदद कर सकता हूँ?"
        elif detected_language == "en-IN":
            return "TAG_ENGLISH I've switched to English. How can I help you?"
    
    # Fast path: Check response cache
    if Config.ENABLE_RESPONSE_CACHE:
        for key, responses in response_cache.items():
            if key in prompt.lower() and detected_language in responses:
                print("Using cached response")
                return responses[detected_language]
    
    # Prepare for LLM call
    lang_tag = "TAG_ENGLISH" if detected_language.startswith("en") else "TAG_HINDI"
    
    # Create messages for LLM
    current_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Prepare API request
    url = "https://api.together.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {together_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "messages": current_messages,
        "max_tokens": 80,
        "temperature": 0.2,
        "top_p": 0.9
    }
    
    # Use a future for timeout management
    try:
        # Make API call with timeout
        response_future = state.executor.submit(
            lambda: requests.post(url, headers=headers, json=data)
        )
        
        # Wait for response with timeout
        response = response_future.result(timeout=Config.LLM_TIMEOUT)
        
        if response.status_code == 200:
            content = response.json().get("choices")[0].get("message").get("content")
            
            # Ensure correct language tag
            if content and not content.startswith(f"TAG_"):
                content = f"{lang_tag} {content}"
                
            return content
        else:
            print(f"API Error: {response.status_code}")
            return get_fallback_response(detected_language)
            
    except Exception as e:
        print(f"LLM timeout or error: {str(e)}")
        return get_fallback_response(detected_language)

# Handle PNR response generation
def handle_pnr_response(pnr_number, language):
    # Get PNR details
    pnr_details = get_pnr_details(pnr_number)
    
    # Generate response based on language
    if language.startswith("hi"):
        return f"TAG_HINDI पीएनआर {pnr_number}: फ्लाइट {pnr_details['flight_no']}, यात्री {pnr_details['passenger_name']}, स्थिति {pnr_details['pnr_status']}, सीट {pnr_details['seat']}।"
    else:
        return f"TAG_ENGLISH PNR {pnr_number}: Flight {pnr_details['flight_no']}, passenger {pnr_details['passenger_name']}, status {pnr_details['pnr_status']}, seat {pnr_details['seat']}."

# Fallback responses when LLM fails
def get_fallback_response(language):
    if language.startswith("hi"):
        return "TAG_HINDI मैं आपकी एयर इंडिया यात्रा में सहायता करने के लिए यहां हूं। कैसे मदद करूं?"
    else:
        return "TAG_ENGLISH I'm here to help with your Air India journey. How can I assist?"

# Fixed text-to-speech
def speak_text(text, language_code):
    # Set flag to avoid concurrent processing
    state.is_synthesizing = True
    
    # Extract clean text (remove tags)
    clean_text = re.sub(r'TAG_\w+\s*', '', text).strip()
    if not clean_text:
        state.is_synthesizing = False
        return
        
    print(f"Speaking ({language_code}): {clean_text}")
    
    # Voice selection based on language
    if language_code.startswith("hi"):
        voice_name = Config.HINDI_VOICE
    else:
        voice_name = Config.ENGLISH_VOICE
    
    # Run TTS in separate thread
    def tts_thread():
        try:
            # Create local synthesizer config with correct voice
            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
            speech_config.speech_synthesis_voice_name = voice_name
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            
            # Speak text
            result = synthesizer.speak_text_async(clean_text).get()
            
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Speech synthesis successful")
            else:
                print(f"Speech synthesis failed: {result.reason}")
                
        except Exception as e:
            print(f"TTS error: {str(e)}")
        finally:
            # Reset flag
            state.is_synthesizing = False
            
            # Delay restart slightly to avoid collision
            time.sleep(0.1)
            
            # Restart recognition only if not in the middle of a language switch
            if not state.switching_language:
                thread = threading.Thread(target=restart_recognition)
                thread.daemon = True
                thread.start()
    
    # Start TTS thread
    thread = threading.Thread(target=tts_thread)
    thread.daemon = True
    thread.start()

# Stop current recognition safely
def stop_recognition():
    try:
        # Only stop if we have an active recognizer
        if state.speech_recognizer and state.recognition_active:
            print("Stopping recognition...")
            state.speech_recognizer.stop_continuous_recognition()
            time.sleep(0.5)  # Wait for stop to complete
            state.recognition_active = False
    except Exception as e:
        print(f"Error stopping recognition: {str(e)}")

# Restart recognition with current language
def restart_recognition():
    try:
        # Only restart if we're not already synthesizing speech
        if not state.is_synthesizing and not state.switching_language:
            print(f"Restarting recognition in {state.current_language}...")
            
            # Complete stop first
            stop_recognition()
            
            # Get recognizer for current language
            recognizer = create_speech_recognizer(state.current_language)
            
            # Setup callbacks
            recognizer.recognized.connect(recognized_callback)
            
            # Start recognition
            recognizer.start_continuous_recognition()
            state.recognition_active = True
            state.speech_recognizer = recognizer
            
            print(f"Now listening in {state.current_language}...")
    except Exception as e:
        print(f"Error restarting recognition: {str(e)}")

# Switch language with proper handling
def switch_language(new_language):
    if state.current_language == new_language:
        return  # Already in the target language
    
    print(f"Switching language from {state.current_language} to {new_language}")
    
    # Set switching flag
    state.switching_language = True
    
    try:
        # Stop the current recognition
        stop_recognition()
        
        # Change language
        state.current_language = new_language
        
        # Create new recognizer for this language
        recognizer = create_speech_recognizer(new_language)
        
        # Setup callbacks
        recognizer.recognized.connect(recognized_callback)
        
        # Start recognition
        recognizer.start_continuous_recognition()
        state.recognition_active = True
        state.speech_recognizer = recognizer
        
        print(f"Now listening in {new_language}")
    except Exception as e:
        print(f"Error during language switch: {str(e)}")
    finally:
        # Reset switching flag
        state.switching_language = False

# Improved recognition callback
def recognized_callback(evt):
    # Ignore if we're already processing or synthesizing
    if state.is_processing or state.is_synthesizing or state.switching_language:
        return
    
    # Check if we have recognized speech
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = evt.result.text.strip()
        
        # Ignore empty or very short input
        if not text or len(text) < 2:
            return
        
        # Set processing flag
        state.is_processing = True
        
        try:
            # Track recognition time
            current_time = time.time()
            recognition_delay = current_time - state.last_recognition_time
            state.last_recognition_time = current_time
            
            print(f"\nUser: '{text}' (processing delay: {recognition_delay:.2f}s)")
            
            # Detect language
            detected_language, confidence = detect_language(text)
            print(f"Language detection: {detected_language} (confidence: {confidence:.2f})")
            
            # Handle language switch if needed
            if confidence >= Config.MIN_CONFIDENCE_THRESHOLD and state.current_language != detected_language:
                # Switch language internally
                old_language = state.current_language
                state.current_language = detected_language
                
                # Process the query directly without language switch acknowledgment
                response = get_llm_response(text, detected_language)
                
                print(f"Assistant: {response}")
                
                # Speak the response
                speak_text(response, detected_language)
                
                # Switch language in a separate thread
                switch_thread = threading.Thread(target=switch_language, args=(detected_language,))
                switch_thread.daemon = True
                switch_thread.start()
                
            else:
                # Process normal input
                response = get_llm_response(text, detected_language)
                print(f"Assistant: {response}")
                
                # Check for special commands
                name_change_match = re.search(r'PROCESS_NAME_CHANGE\s*([^;]+);([^;]+);(.+)', response)
                
                if name_change_match:
                    # Process name change
                    pnr = name_change_match.group(1).strip()
                    old_name = name_change_match.group(2).strip()
                    new_name = name_change_match.group(3).strip()
                    
                    process_name_change(pnr, old_name, new_name)
                    
                    # Generate confirmation based on language
                    if detected_language.startswith("hi"):
                        confirm = "TAG_HINDI नाम बदलने का अनुरोध सफल हुआ। हमारी टीम जल्द संपर्क करेगी।"
                    else:
                        confirm = "TAG_ENGLISH Name change request successful. Our team will contact you soon."
                        
                    # Speak confirmation
                    speak_text(confirm, detected_language)
                else:
                    # Normal response
                    speak_text(response, detected_language)
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
            # Provide error message in current language
            error_msg = "TAG_HINDI क्षमा करें, त्रुटि हुई।" if detected_language.startswith("hi") else "TAG_ENGLISH Sorry, an error occurred."
            speak_text(error_msg, detected_language)
        finally:
            # Reset processing flag
            state.is_processing = False
            
# Play introduction message
def play_introduction():
    if state.introduction_played:
        return
    
    # Use shorter introduction
    intro_text = introduction_messages[Config.DEFAULT_LANGUAGE]
    print(f"Playing introduction...")
    print(f"Assistant: {intro_text}")
    
    # Speak introduction
    speak_text(intro_text, Config.DEFAULT_LANGUAGE)
    state.introduction_played = True

# Main function
def main():
    print("\n" + "="*60)
    print("AIR INDIA VOICE ASSISTANT - FIXED VERSION")
    print("="*60)
    print("* Fast response multilingual support (Hindi/English)")
    print("* Instant language switching")
    print("* Quick PNR status lookup")
    print("* Optimized for natural conversation")
    print("="*60 + "\n")
    
    # Setup speech configuration
    setup_speech_config()
    
    # Initialize speech recognizer with default language
    recognizer = create_speech_recognizer(Config.DEFAULT_LANGUAGE)
    state.speech_recognizer = recognizer
    
    # Setup callbacks
    recognizer.recognized.connect(recognized_callback)
    
    # Start recognition
    recognizer.start_continuous_recognition()
    state.recognition_active = True
    
    # Play introduction
    play_introduction()
    
    # Keep program running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_recognition()
        state.executor.shutdown()
        print("Voice assistant terminated.")

if __name__ == "__main__":
    main()
