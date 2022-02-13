from skimage.transform import resize
import numpy as np
import functools

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from beheaded_inception3 import beheaded_inception_v3


def greedy_sample(logits):
    return logits.argmax(dim=-1)


def random_sample(logits, t):
    probabilities = F.softmax(logits / t, dim=-1)
    distribution = Categorical(probabilities)
    return distribution.sample()


def prepare_img_for_classifier_backbone(image, device):
    image = resize(image, (299, 299))
    image = torch.tensor(image.transpose([2, 0, 1])[None], dtype=torch.float32, device=device)
    return image


def prepare_img_for_detector_backbone(image, device, detection_transform):
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = detection_transform([image])[0].tensors.to(device)
    return image


class ImageProcessor:
    def __init__(self, vocabulary, caption_net, model_path, with_attention, prepare_image, device=None):
        self.vocabulary = vocabulary
        self.with_attention = with_attention
        self.prepare_image = prepare_image

        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.caption_net = caption_net
        if not with_attention:
            self.inception = beheaded_inception_v3().to(self.device).train(False)
        self.caption_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.caption_net.to(self.device).eval()

    def describe_image(self, image, sample, t=0.5, max_len=100):
        image = self.prepare_image(image, self.device)
        with torch.no_grad():
            if not self.with_attention:
                _, vectors_neck, _ = self.inception(image)
                generated = [self.vocabulary.find_token_index('<sos>')]

                # слово за словом генерируем описание картинки
                for _ in range(max_len):
                    caption = torch.tensor([generated], device=self.device)
                    logits = self.caption_net(vectors_neck, caption)[:, -1]

                    if sample:
                        idx = random_sample(logits, t).item()
                    else:
                        idx = greedy_sample(logits).item()
                    generated.append(idx)

                    if idx == self.vocabulary.find_token_index('<eos>'):
                        break
            else:
                if sample:
                    generated, _, _ = self.caption_net(image, None, sample=functools.partial(random_sample, t=t),
                                                       max_len=max_len)
                else:
                    generated, _, _ = self.caption_net(image, None, sample=greedy_sample, max_len=max_len)
                generated = generated[0].cpu().detach().numpy()

            generated = [self.vocabulary[idx] for idx in generated]
            return ' '.join(generated[1:-1])
