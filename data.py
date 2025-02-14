import os
from utils import save_pkl, load_pkl

class Multi30k():
    def __init__(self,
                 lang=("en", "de")):
        self.dataset_name = "multi30k"
        self.lang_src, self.lang_tgt = lang

        self.tokenizer_src = self.build_tokenizer(self.lang_src)
        self.tokenizer_tgt = self.build_tokenizer(self.lang_tgt)

        self.train = None
        self.valid = None
        self.test = None
        self.build_dataset()

    def build_dataset(self, raw_dir="raw", cache_dir=".data"):
        cache_dir = os.path.join(cache_dir, self.dataset_name)
        raw_dir = os.path.join(cache_dir, raw_dir)
        os.makedirs(raw_dir, exist_ok=True)

        train_file = os.path.join(cache_dir, "train.pkl")
        valid_file = os.path.join(cache_dir, "valid.pkl")
        test_file = os.path.join(cache_dir, "test.pkl")

        if os.path.exists(train_file):
            self.train = load_pkl(train_file)
        else:
            with open(os.path.join(raw_dir, "train.en"), "r") as f:
                train_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "train.de"), "r") as f:
                train_de = [text.rstrip() for text in f]
            self.train = [(en, de) for en, de in zip(train_en, train_de)]
            save_pkl(self.train, train_file)

        if os.path.exists(valid_file):
            self.valid = load_pkl(valid_file)
        else:
            with open(os.path.join(raw_dir, "val.en"), "r") as f:
                valid_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "val.de"), "r") as f:
                valid_de = [text.rstrip() for text in f]
            self.valid = [(en, de) for en, de in zip(valid_en, valid_de)]
            save_pkl(self.valid, valid_file)

        if os.path.exists(test_file):
            self.test = load_pkl(test_file)
        else:
            with open(os.path.join(raw_dir, "test_2016_flickr.en"), "r") as f:
                test_en = [text.rstrip() for text in f]
            with open(os.path.join(raw_dir, "test_2016_flickr.de"), "r") as f:
                test_de = [text.rstrip() for text in f]
            self.test = [(en, de) for en, de in zip(test_en, test_de)]
            save_pkl(self.test, test_file)

    def build_tokenizer(self, lang):
        from torchtext.data.utils import get_tokenizer
        spacy_lang_dict = {
            'en': "en_core_web_sm",
            'de': "de_core_news_sm"
        }
        assert lang in spacy_lang_dict.keys()
        return get_tokenizer("spacy", spacy_lang_dict[lang])
