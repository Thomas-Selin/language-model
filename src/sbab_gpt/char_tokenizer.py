import json

class CharTokenizer:
    def __init__(self, chars_file="chars.json"):
        with open(chars_file, 'r') as f:
            self.chars = json.load(f)
        
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

# Save the character set
if __name__ == "__main__":
    from gpt import chars
    with open("hf_export/chars.json", "w") as f:
        json.dump(chars, f)
    print("Character set saved to hf_export/chars.json")
