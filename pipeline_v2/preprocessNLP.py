import re
import underthesea
import unicodedata


class Preprocessor:
    def __init__(self, stopwords_file="../dataset/vietnamese-stopwords.txt", remove_stopwords=False, lemmatization=True):
        self.stopwords = self.load_stopwords(stopwords_file)
        self.lemmatizer_dict = {
            "k": "không",
            "ko": "không",
            "hok": "không",
            "khg": "không",
            "thik": "thích",
            "thjk": "thích",
            "ng": "người",
            "hok_tốt": "không_tốt",
            "bth": "bình_thường",
            "bt": "bình_thường",
            "ok": "tốt",
            "okie": "tốt",
            "tks": "cảm_ơn",
            "thank": "cảm_ơn",
            "thanks": "cảm_ơn",
            "tạm_được": "tạm_được",
            "tạm_ok": "tạm_được",
        }
        self.remove_stopwords = remove_stopwords
        self.lemmatization = lemmatization
        self.whitelist = { "tốt", "xấu", "hay", "dở", "giỏi", "kém", "tuyệt", "tuyệt vời", "kinh khủng", "chán", "dễ", "khó", "nhanh", "chậm", "đẹp", "xấu xí", "xấu", "vui", "yêu", "buồn", "ghét", "thích", "thơm", "ưng ý", "bền", "chắc", "chắc chắn", "sạch", "bẩn", "rẻ", "đắt", "mạnh", "yếu", "to", "nhỏ", "mềm", "cứng", "mượt", "ngon", "dở tệ", "tệ", "tệ hại", "hỏng", "lỗi", "chất lượng", "mượt mà", "vừa", "ổn", "đúng", "chuẩn"}

    def load_stopwords(self, file_path):
        stopwords = set()
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().lower()
                word = unicodedata.normalize("NFC", word)
                if word:
                    stopwords.add(word)
        return stopwords

    def lemmatizer(self, tokens):
        return [t if t not in self.lemmatizer_dict else self.lemmatizer_dict[t] for t in tokens]
    
    def lower_strip(self, text):
        return text.lower().strip()
    
    def normalize(self, text):
        text = underthesea.text_normalize(text)
        text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", text)
        return text

    def tokenize(self, text):
        return underthesea.word_tokenize(text)
    
    def remove_stopword(self, tokens):
        return [t for t in tokens if (t.strip().lower() not in self.stopwords) or (t.strip().lower() in self.whitelist)]
    
    def remove_outliers(self, text):
        text = text.split()
        if len(text) > 27:
            return ""
        return " ".join(text)
    

    def preprocess_text(self, text):

        # Loại bỏ outliers
        text = self.remove_outliers(text)
        if not text:
            return ""

        # Chuẩn hóa: về chữ thường
        text = self.lower_strip(text)
        #print(text)
        
        # Xoá ký tự đặc biệt, giữ lại chữ và số
        # Chuẩn hóa unicode
        text = self.normalize(text)
        
        # Tokenize (dùng underthesea)
        tokens = self.tokenize(text)
        # print(tokens)
        
        # Lemmatization (theo dictionary mapping)
        if self.lemmatization:
            tokens = self.lemmatizer(tokens)
        # print(tokens)
        
        # Loại bỏ stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopword(tokens)


        # Nối lại chuỗi
        tokens = " ".join(tokens).strip()

        # Tokenize lại để chuẩn hóa dấu gạch dưới
        tokens = underthesea.word_tokenize(tokens, format="text")

        return str(tokens)
    