from subword_tokenizer import SubwordTokenizer
tok = SubwordTokenizer(vocab_file="data/output/vocab_subword.json")
s = "There was a psychedelic rabbit, jumping over 3 fences."
print(tok.encode(s))
print(tok.decode(tok.encode(s)))