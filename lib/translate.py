import requests
import langid
def detect_language(input_text):
    lang, confidence = langid.classify(input_text)
    print(lang)
    return lang

def translate_text(input_text, source_language, target_language):
    api_url = f"https://mymemory.translated.net/api/get?langpair={source_language}|{target_language}&q={input_text}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()  
        
        translation = response.json()["responseData"]["translatedText"]
        return translation
    except requests.exceptions.RequestException as e:
        print(f"Translation request failed: {e}")
        return None
