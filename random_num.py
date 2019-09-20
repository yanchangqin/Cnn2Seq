import numpy as np
from PIL import Image,ImageDraw,ImageFont

font = ImageFont.truetype('1.ttf',size=36)
font1 = ImageFont.truetype('2.ttf',size=36)
font2 = ImageFont.truetype('3.ttf',size=36)
font3 = ImageFont.truetype('4.ttf',size=36)
def num():
    return chr(np.random.randint(48,57))

def forward_color():
    return (np.random.randint(120,200),
           np.random.randint(120,200),
           np.random.randint(120,200))

def back_color():
    return (np.random.randint(50, 150),
            np.random.randint(50, 150),
            np.random.randint(50, 150))

w = 120
h =60

path = r'code1'
for i in range(200):
    image =Image.new('RGB',(w,h),color=(255,255,255))
    img = ImageDraw.Draw(image)
    for j in range(w):
        for k in range(h):
            img.point((j,k),fill=back_color())
    list_num = []
    for m in range(4):
        ch =num()
        list_num.append(ch)
        img.text((30*m+10,18),text=ch,fill=forward_color(),font=font1)
    image.save('{0}/{1}.jpg'.format(path,''.join(list_num)))
