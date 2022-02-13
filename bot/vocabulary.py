import json


class Vocabulary:
    def __init__(self, idx2token_path):
        with open(idx2token_path, 'r') as f:
            self.idx2token = json.load(f)
            self.idx2token = dict(map(lambda item: (int(item[0]), item[1]), self.idx2token.items()))

    def find_token_index(self, token):
        iterator = filter(lambda item: item[1] == token, self.idx2token.items())
        return next(iterator)[0]

    def __getitem__(self, item):
        return self.idx2token[item]

    def __len__(self):
        return len(self.idx2token)
