# -*- coding: utf-8 -*-

import sys

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

def read_large_file(file_handler, block_size=100000):
    block = []
    for line in file_handler:
        block.append(line)
        if len(block) == block_size:
            yield block
            block = []
    # don't forget to yield the last block
    if block:
        yield block

puncts = u""".,",:,),(,!,?,|,;,$,&,/,[,],>,%,=,#,*,+,\,•,~,@,£,·,_,{,},©,^,®,`,<,→,°,€,™,›,♥,←,×,§,″,′,Â,█,½,à,…,“,★,”,–,●,â,►,−,¢,²,¬,░,¶,↑,±,¿,▾,═,¦,║,―,¥,▓,—,‹,─,▒,：,¼,⊕,▼,▪,†,■,’,▀,¨,▄,♫,☆,¯,♦,¤,▲,è,¸,¾,Ã,⋅,‘,∞,∙,）,↓,、,│,（,»,，,♪,╩,╚,³,・,╦,╣,╔,╗,▬,❤,ï,Ø,¹,≤,‡,√""".split(',')
puncts = puncts + [u',',u'-',u"'"]
# keep é

puncts = [x for x in puncts if x != '']
stopwords = ['','s','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
# stopwords=['','s']
stopwords = set(stopwords)

def clean_text(x):
    for punct in puncts:
        x = x.replace(punct, ' ')
    return x

def split_text(text):
    if text=='__EMPTY__':
        return text.split(' ')
    
    x = clean_text(text)
    x = x.lower()
    x = x.split(' ')
    x = [w for w in x if w not in stopwords]
    if len(x)==0 or x == ['']:
        x = ['__EMPTY__']
        print('null',text)
    return x

def get_split_text(x):
    return ' '.join(split_text(x))

def get_sorted_split_text(x):
    return ' '.join(sorted(split_text(x)))


def get_trichar(words):
    trichars = []
    for word in words:
        word = "#"+word+"#"
        for i in range(len(word)-2):
            trichar = word[i:i+3]
    
            trichars.append(trichar)
    
    return trichars


valid_test_lastword = {'accessories',
 'agent',
 'animal',
 'anklet',
 'antenna',
 'apron',
 'ark',
 'ashtray',
 'backpack',
 'bag',
 'bags',
 'balloon',
 'balm',
 'basin',
 'basket',
 'beans',
 'bed',
 'belt',
 'bibs',
 'bikini',
 'biscuits',
 'blanket',
 'blocks',
 'blouse',
 'boots',
 'bottle',
 'box',
 'bracelet',
 'brooch',
 'brush',
 'brushes',
 'cabinet',
 'caddy',
 'cap',
 'car',
 'carpet',
 'case',
 'cases',
 'ceiling',
 'censer',
 'chair',
 'chandelier',
 'charger',
 'cheongsam',
 'child',
 'chocolate',
 'chopsticks',
 'clip',
 'clock',
 'clothes',
 'clothesline',
 'clothing',
 'coat',
 'collar',
 'comb',
 'conditioning',
 'control',
 'cord',
 'corsage',
 'costume',
 'cotton',
 'covers',
 'crib',
 'cube',
 'cuff',
 'cufflinks',
 'cup',
 'cups',
 'curtain',
 'dance',
 'desk',
 'device',
 'dispenser',
 'drawers',
 'dress',
 'earrings',
 'enclosures',
 'eyelashes',
 'fan',
 'filter',
 'frame',
 'furniture',
 'gear',
 'glass',
 'gloves',
 'goggles',
 'gyro',
 'hammock',
 'handbag',
 'hat',
 'headband',
 'headdress',
 'headrest',
 'holder',
 'hood',
 'hoodie',
 'humidifier',
 'ink',
 'insole',
 'instruments',
 'inverter',
 'jacket',
 'jeans',
 'johns',
 'kettle',
 'keychain',
 'knit',
 'lantern',
 'leather',
 'leggings',
 'lens',
 'lid',
 'light',
 'lights',
 'linen',
 'lock',
 'low',
 'lumbar',
 'machine',
 'marker',
 'mask',
 'masks',
 'mats',
 'mirror',
 'mold',
 'mug',
 'necklace',
 'nets',
 'nightgown',
 'opener',
 'ornaments',
 'pack',
 'package',
 'pad',
 'painting',
 'pants',
 'paper',
 'pen',
 'pencil',
 'perfume',
 'pet',
 'phone',
 'pillow',
 'plants',
 'po',
 'pockets',
 'pole',
 'pots',
 'power',
 'pump',
 'purifier',
 'purse',
 'quilt',
 'racket',
 'raincoat',
 'rod',
 'sandals',
 'sauce',
 'scarf',
 'schoolbag',
 'scooter',
 'seal',
 'seaweed',
 'sensor',
 'sets',
 'shaver',
 'shell',
 'shirt',
 'shoe',
 'shoes',
 'shovel',
 'skirt',
 'slippers',
 'socket',
 'socks',
 'sofa',
 'spoon',
 'spotlights',
 'steamer',
 'stickers',
 'stitch',
 'strip',
 'suit',
 'sunglasses',
 'sweater',
 'sweatshirt',
 'swimsuit',
 'talkie',
 'teapot',
 'thread',
 'tie',
 'toothpaste',
 'top',
 'towel',
 'tracksuit',
 'trash',
 'tray',
 'tricycle',
 'trousers',
 'tureen',
 'tv',
 'umbrella',
 'underwear',
 'vase',
 'vest',
 'wallet',
 'wardrobe',
 'watch',
 'wear',
 'wedding',
 'windbreaker',
 'wipes'}


def lastword_filter(text):
    if split_text(text)[-1] in valid_test_lastword:
        return True
    else:
        return False