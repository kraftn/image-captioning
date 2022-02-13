import logging
import sys
import io
from PIL import Image
import numpy as np
import functools
from torchvision import models
import os
import gdown

from telegram import Update
from telegram.ext import Updater, CallbackContext, Filters
from telegram import ext

from vocabulary import Vocabulary
from attention_caption_net import FasterRCNNEncoder, Decoder, Attention, MultipleAttention, CaptionNetWithAttention
from caption_net import CaptionNet
from beheaded_inception3 import beheaded_inception_v3
from image_processor import ImageProcessor, prepare_img_for_classifier_backbone, prepare_img_for_detector_backbone

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def answer_to_start(update: Update, context: CallbackContext):
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    context.bot.send_message(chat_id=update.message.chat_id, text='Чтобы получить описание, отправьте изображение в чат.')
    context.bot.send_message(chat_id=update.message.chat_id, text='Если необходимо сгенерировать несколько вариантов описания для одного и того же изображения, отправьте вместе с фото необходимое число. Максимум описаний за раз - 10.')


def process_image(update: Update, context: CallbackContext, image_processor: ImageProcessor):
    bot = context.bot
    bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    message = update.message
    attachment = message.effective_attachment

    if isinstance(attachment, list):
        file = bot.get_file(sorted(attachment, key=lambda x: x.file_size)[-1].file_id)
    else:
        if attachment.mime_type.startswith('image'):
            file = bot.get_file(attachment.file_id)
        else:
            bot.send_message(chat_id=update.effective_chat.id, text='Файл не является изображением',
                             reply_to_message_id=message.message_id)
            return

    image_bytes = file.download_as_bytearray()
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB'))

    try:
        n_times = min(10, int(message.caption))
    except (TypeError, ValueError):
        n_times = 1

    answer = []
    for i in range(n_times):
        answer.append(image_processor.describe_image(image, sample=True, t=0.35))
    bot.send_message(chat_id=update.effective_chat.id, text='\n'.join(answer), reply_to_message_id=message.message_id)


def error(update, context):
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def download_model(model_file_id, idx2token_file_id):
    if not os.path.exists('./data'):
        os.mkdir('./data')
    gdown.download(id=model_file_id, output='./data/best_model.pt', use_cookies=False)
    gdown.download(id=idx2token_file_id, output='./data/idx2token.json', use_cookies=False)


def main():
    download_model('1NrfGJneK_PmUoqWaVVARsD6HmszMqbLI', '1GGFXFqpncL1YKmj2YkiUiGRGvYTF_uNz')

    vocabulary = Vocabulary('./data/idx2token.json')
    fasterrcnn = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained_backbone=False, pretrained=False).eval()
    caption_net = CaptionNetWithAttention(encoder=FasterRCNNEncoder(fasterrcnn.backbone).eval(),
                                          decoder=Decoder(vocab_size=len(vocabulary), embedding_size=256,
                                                          dec_hid_size=1024, num_layers=2, dropout=0.5,
                                                          attention_layer=MultipleAttention([256, 256, 256], 1024),
                                                          skip_connections=True,
                                                          padding_idx=vocabulary.find_token_index('<pad>')),
                                          init_hidden_cell=True,
                                          out_enc_hid_size=256,
                                          dropout=0.5,
                                          sos=vocabulary.find_token_index('<sos>'),
                                          eos=vocabulary.find_token_index('<eos>'))
    prepare_image = functools.partial(prepare_img_for_detector_backbone, detection_transform=fasterrcnn.transform)
    image_processor = ImageProcessor(vocabulary, caption_net, './data/best_model.pt', with_attention=True,
                                     prepare_image=prepare_image, device='cpu')

    bot_token = sys.argv[1]
    updater = Updater(token=bot_token, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_error_handler(error)

    start_handler = ext.CommandHandler('start', answer_to_start)
    dispatcher.add_handler(start_handler)

    photo_handler = ext.MessageHandler(Filters.attachment,
                                       functools.partial(process_image, image_processor=image_processor))
    dispatcher.add_handler(photo_handler)

    if sys.argv[2] == 'webhook':
        port = int(os.environ.get('PORT', '8443'))
        updater.start_webhook(listen='0.0.0.0', port=port, url_path=bot_token,
                              webhook_url=f'https://{sys.argv[3]}.herokuapp.com/{bot_token}')
    elif sys.argv[2] == 'polling':
        updater.start_polling()
    del fasterrcnn
    updater.idle()


if __name__ == '__main__':
    main()
