from pathlib import Path
from io import BytesIO
FOLDER=Path(__file__).parent

def label(text,position,size=(.13,.026),color=(.96,.97,.98),subtitle=None):
    from PIL import Image,ImageDraw,ImageFont
    w,h=size;width,height=round(w*12000),round(h*12000)
    image=Image.new('RGB',(width,height),tuple(round(c*255) for c in color));draw=ImageDraw.Draw(image)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    font=ImageFont.truetype(font_path,height)
    l,t,r,b=draw.textbbox((0,0),text,font=font)
    font=ImageFont.truetype(font_path,max(1,int(height*min(.90*width/(r-l),(.45 if subtitle else .74)*height/(b-t)))))
    l,t,r,b=draw.textbbox((0,0),text,font=font)
    centre=.34*height if subtitle else .5*height
    draw.text(((width-r-l)/2,centre-(b+t)/2),text,font=font,fill=(10,16,22))
    if subtitle:
        small=ImageFont.truetype(font_path,round(.21*height))
        l,t,r,b=draw.textbbox((0,0),subtitle,font=small)
        draw.text(((width-r-l)/2,.79*height-(b+t)/2),subtitle,font=small,fill=(10,16,22))
    draw.rectangle((1,1,width-2,height-2),outline=(55,65,75),width=2)
    folder=FOLDER/'assets';folder.mkdir(exist_ok=True)
    name=''.join(c if c.isalnum() else '_' for c in text)+'.png';path=folder/name
    data=BytesIO();image.save(data,format='PNG')
    if not path.exists() or path.read_bytes()!=data.getvalue():path.write_bytes(data.getvalue())
    return dict(type='mesh',position=list(position),collision=False,roughness=1.,texture='assets/'+name,
        mesh=dict(points=[[-h/2,-w/2,0],[-h/2,w/2,0],[h/2,w/2,0],[h/2,-w/2,0]],
                  faces=[[0,3,2,1]],uv=[[1,0],[1,1],[0,1],[0,0]]))


def capsule(a,b,r,color,**kw):
    return dict(type='capsule',size=[r,0.],fromto=[*a,*b],color=list(color),**kw)
